from flask import request, render_template, redirect, url_for, jsonify, current_app
import os
import tarfile
from functools import partial


from uuid import uuid4
from app.globals import get_db
from app.utils.s3 import s3_upload, s3_download, get_s3_url

ALLOWED_EXTENSIONS = ['csv', 'json']
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


def rate(dataset_name):
    label = request.args.get("label")
    id = request.args.get("image_id")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE images SET label=%s WHERE dataset_name=%s and id=%s", (label, dataset_name, id))
    cursor.close()
    conn.commit()
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

def view(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, path FROM images WHERE dataset_name = %s and label ISNULL LIMIT 1", (dataset_name,))
    res = cursor.fetchone()
    if res:
        id, path = res
        cursor.execute("SELECT category FROM categories WHERE dataset_name = %s", (dataset_name,))
        categories = [tup[0] for tup in cursor.fetchall()]
        cursor.close()
        return render_template('dataset.html', name=dataset_name, image=path, categories=categories, image_id=id)
    return "Dataset does not exist", 404

