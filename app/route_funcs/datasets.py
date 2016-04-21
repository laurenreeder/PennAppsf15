from flask import request, render_template, redirect, url_for, jsonify, current_app
import os
import tarfile
from functools import partial
from cStringIO import StringIO
from ml import svm
import threading
from mturk import create_hit as mt
from uuid import uuid4
from app.globals import get_db, local_connect
from app.utils.s3 import s3_upload, s3_download, get_s3_url, s3_upload_images

ALLOWED_EXTENSIONS = ['tar', 'zip']
shortcuts = ["A", "S", "D", "F", "Space", "J", "K", "L", ";"]

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def unzipFile(fileName, dirName):
    extract_dir = "./static/images/%s" % dirName
    os.mkdir(extract_dir)
    tf = tarfile.open(name=fileName)
    tf.extractall(path=extract_dir)

    return [extract_dir + "/" + member.name for member in tf.getmembers() if member.isfile() and member.name.split('/')[-1][0] != "."]

def bg_train(dataset_name):
    bg_thread = threading.Thread(target=partial(update_model, dataset_name, local_connect()))
    bg_thread.start()


def rate(dataset_name):
    label = request.args.get("label")
    id = request.args.get("image_id")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE images SET label=%s WHERE dataset_name=%s and id=%s", (label, dataset_name, id))
    cursor.execute("UPDATE datasets SET model_updated = false WHERE name=%s",  (dataset_name,))
    cursor.execute("SELECT count(*) from images WHERE dataset_name=%s and label NOTNULL", (dataset_name,))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.commit()
    if count % 20 == 0:
        bg_train(dataset_name)

    return redirect(url_for('datasets_view', dataset_name=dataset_name))

def new():
    if request.method == "POST":
        name = request.form['dataset_name']
        
        print request.form
        categories = request.form.getlist('category')
        s3_key = None
        print request.files
        #if request.form.get('dataset_url', False):
        #    s3_key = s3_download(request.form['dataset_url'])

        if 'dataset_upload' in request.files:
            dataset = request.files['dataset_upload']
            if dataset:
                id = uuid4().hex
                filename = dataset.filename
                path_to_file = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(current_app.config['UPLOAD_FOLDER']):
                    os.mkdir(current_app.config['UPLOAD_FOLDER'])
                dataset.save(path_to_file)
                results = unzipFile(path_to_file, str(id))
                print results
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO datasets VALUES (%s,%s)", (name, s3_key))
                for path in results:
                    image_id = uuid4().hex
                    cursor.execute("INSERT INTO images VALUES (%s,%s,%s)", (image_id, name, path[1:]))
                for category in categories:
                    cursor.execute("INSERT INTO categories VALUES (%s,%s)", (category, name))
                s3_thread = threading.Thread(target=partial(s3_upload_images, results, current_app.config["S3_KEY"], current_app.config["S3_SECRET"], current_app.config["S3_BUCKET"]))
                s3_thread.start()
                conn.commit()
                cursor.close()
                return redirect(url_for('datasets_view', dataset_name=name))

    return render_template('dataset_upload.html')

def test():
    categories = ["Apple", "Orange", "Erik"]
    mapping = {}

    i = 0
    for category in categories:
        mapping[category] = shortcuts[i]
        i = i + 1
    return render_template('dataset.html', name="dataset_name", image="../static/img/mountain.jpg", categories=categories, mapping=mapping)

def train(dataset_name, db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT path, label FROM images WHERE dataset_name = %s and label NOTNULL", (dataset_name,))
    results = cursor.fetchall()
    cursor.close()
    image_paths = ["." + r[0] for r in results]
    image_labels = [r[1] for r in results]
    clf = svm.train_with_images(image_paths, image_labels)
    print svm.test_with_images(clf, image_paths, image_labels)
    return clf

def update_model(dataset_name, conn):
    clf = train(dataset_name, conn)
    cursor = conn.cursor()
    cursor.execute("SELECT id, path FROM images WHERE dataset_name = %s and label ISNULL", (dataset_name,))
    results = cursor.fetchall()
    paths = ["." + result[1] for result in results]
    ids = [result[0] for result in results]
    feature_vecs = svm.get_features(paths)
    decs = clf.decision_function(feature_vecs)
    predictions = clf.predict(feature_vecs)
    dec_vals = map(lambda dists: sum(map(abs, dists)), decs)
    update_data = StringIO('\n'.join(map(lambda tup: tup[0] + '\t' + tup[1] + '\t' + str(tup[2]),
                                         zip(ids, predictions, dec_vals))))
    cursor.execute("CREATE TEMPORARY TABLE model_updates (id varchar, prediction varchar, dist float) ON COMMIT DROP")
    cursor.copy_from(update_data, 'model_updates')
    cursor.execute("UPDATE images i SET dist_from_surface = mu.dist, prediction = mu.prediction FROM model_updates mu WHERE i.id = mu.id")
    cursor.execute("UPDATE datasets SET model_updated = true WHERE name = %s", (dataset_name,))
    cursor.close()
    conn.commit()
    conn.close()
    curr_updating.remove(dataset_name)
    print "trained"
    return zip(paths, map(lambda dists: sum(map(abs, dists)), decs))

def mturk(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM images WHERE dataset_name = %s", (dataset_name,))
    image_paths = cursor.fetchall()
    cursor.execute("SELECT category FROM categories WHERE dataset_name = %s", (dataset_name,))
    categories = cursor.fetchall()
    HIT_ids = []
    for image_path in image_paths:
        s3_url = get_s3_url(image_path[0])        
        HIT_ids.append(mt.create_hit(s3_url, [c[0] for c in categories]))

    return render_template('mturk_results.html')

def view(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, path FROM images WHERE dataset_name = %s and label ISNULL ORDER BY dist_from_surface ASC LIMIT 1",
                   (dataset_name,))
    res = cursor.fetchone()
    if res:
        id, path = res
        cursor.execute("SELECT category FROM categories WHERE dataset_name = %s", (dataset_name,))
        categories = [tup[0] for tup in cursor.fetchall()]
        cursor.close()

        mapping = {}

        i = 0
        for category in categories:
            mapping[category] = shortcuts[i]
            i = i + 1

        return render_template('dataset.html', name=dataset_name, image=path, categories=categories, image_id=id, mapping=mapping)
    return "Dataset does not exist", 404


def model_is_updated(db_conn, dataset_name):
    cursor = db_conn.cursor()
    cursor.execute("SELECT model_updated from datasets WHERE name = %s", (dataset_name, ))
    updated = cursor.fetchone()[0]
    cursor.close()
    return updated


curr_updating = set()
def get_predictions(dataset_name):
    conn = get_db()
    if model_is_updated(conn, dataset_name):
        cursor = conn.cursor()
        cursor.execute("SELECT path, prediction FROM images WHERE dataset_name = %s and label ISNULL ORDER BY prediction", (dataset_name,))
        results = cursor.fetchall()
        predictions = {}
        for path, prediction in results:
            if prediction in predictions:
                predictions[prediction] += [path]
            else:
                predictions[prediction] = [path]
        print "predictions:", predictions
        return jsonify({"updated": True,
                        "prediction_html": render_template('learning_results.html', labeled_images=predictions)})
    else:
        if dataset_name not in curr_updating:
            curr_updating.add(dataset_name)
            bg_train(dataset_name)

        return jsonify({"updated": False})




def results(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT path, label FROM images WHERE dataset_name = %s and label NOTNULL ORDER BY label", (dataset_name,))
    results = cursor.fetchall()
    labeled_images = {}

    for path, label in results:
        if label in labeled_images:
            labeled_images[label] = labeled_images[label] + [path]
        else:
            labeled_images[label] = [path]

    return render_template('learning.html', labeled_images=labeled_images, labels=labeled_images.keys(), name=dataset_name)

