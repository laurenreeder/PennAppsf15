from flask import request, render_template, redirect, url_for, jsonify
from functools import partial


from uuid import uuid4
from app.globals import get_db
from app.utils.s3 import s3_upload, s3_download, get_s3_url
from app.mixed_models import models_by_task, run_file

ALLOWED_EXTENSIONS = ['csv', 'json']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def new():
    if request.method == "POST":
        name = request.form['dataset_name']
        print request.form
        s3_key = None
        print request.files
        if request.form.get('dataset_url', False):
            s3_key = s3_download(request.form['dataset_url'])

        elif 'dataset' in request.files:
            dataset = request.files['dataset']
            print dataset
            if dataset:
                s3_key = s3_upload(dataset)

        if s3_key:
            print s3_key
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO datasets VALUES (%s,%s)", (name, s3_key))
            conn.commit()
            cursor.close()
            return redirect(url_for('datasets_view', dataset_name=name))

    return render_template('dataset_upload.html')

def view(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT s3_key FROM datasets WHERE name = %s LIMIT 1", (dataset_name,))
    res = cursor.fetchone()
    cursor.close()
    if res is not None:
        s3_key = res[0]
        s3_url = get_s3_url(s3_key)
        return render_template('dataset.html', download_url=s3_url, s3_key=s3_key, name=dataset_name, image="../static/img/mountain.jpg", categories=["Apple", "Orange", "Erik"])
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

