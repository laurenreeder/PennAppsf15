from flask import request, render_template, redirect, url_for, jsonify, current_app
import os
import tarfile
from functools import partial


from uuid import uuid4
from app.globals import get_db
from app.utils.s3 import s3_upload, s3_download, get_s3_url
from app.mixed_models import models_by_task, run_file

ALLOWED_EXTENSIONS = ['csv', 'json']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def unzipFile(fileName, dirName):
    extract_dir = "./static/images/%s" % dirName
    os.mkdir(extract_dir)

    tf = tarfile.open(name=fileName)
    tf.extractall(path=extract_dir)
    return [extract_dir + "/" + member.name for member in tf.getmembers() if member.isfile() and not member.name.split('/')[1].startswith("._")]


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
                    cursor.execute("INSERT INTO images VALUES (%s,%s)", (name, path))
                for category in categories:
                    cursor.execute("INSERT INTO categories VALUES (%s,%s)", (name, category))
                conn.commit()
                cursor.close()
                return redirect(url_for('datasets_view', dataset_name=name))

    return render_template('dataset_upload.html')

def test():

    return render_template('dataset.html', name="dataset_name", image="../static/img/mountain.jpg", categories=["Apple", "Orange", "Erik"])

def view(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM images WHERE dataset_name = %s LIMIT 1", (dataset_name,))
    res = cursor.fetchone()[0][1:]
    cursor.execute("SELECT category FROM categories WHERE dataset_name = %s", (dataset_name,))
    categories = [tup[0] for tup in cursor.fetchall()]
    cursor.close()
    if res is not None:
        #s3_key = res[0]
        #s3_url = get_s3_url(s3_key)
        return render_template('dataset.html', name=dataset_name, image=res, categories=categories)
    return "Dataset does not exist", 404

def learn(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT s3_key FROM datasets WHERE name = %s LIMIT 1", (dataset_name,))
    res = cursor.fetchone()
    cursor.close()
    if res is not None:
        s3_key = res[0]
        return render_template('learning.html', s3_key=s3_key, models_by_task=models_by_task)
    return "Dataset does not exist", 404

from multiprocessing.pool import Pool
pool = Pool(processes=2)

results = {}
def set_results(code, result):
    global results
    results[code] = result


def run_learning():
    s3_key = request.form['s3_key']
    model_type = request.form['model_type']
    model_name = request.form['model']
    code = uuid4().hex
    global results
    results[code] = None
    async_result = pool.apply_async(run_file, (s3_key, model_name, model_type), callback=partial(set_results, code))
    run_file(s3_key, model_name, model_type)
    return 

def get_learning_result():
    code = request.args['result_id']
    if results[code] is not None:
        if type(results[code]) in (list, dict):
            return jsonify(results[code])
        else:
            return results[code]
    else:
        return "Unfinished"

