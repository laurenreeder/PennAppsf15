from flask import request, render_template, redirect, url_for

from app.globals import get_db
from app.utils.s3 import s3_upload, get_s3_url

ALLOWED_EXTENSIONS = ['csv', 'json']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def new():
    dataset = request.files['dataset']
    name = request.form['name']
    if dataset and allowed_file(dataset.filename):
        filename = s3_upload(dataset)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO datasets VALUES (%s,%s)", (name, filename))
        conn.commit()
        cursor.close()
        return redirect(url_for('datasets_view', dataset_name=filename))
    return redirect(url_for('index'))

def view(dataset_name):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT s3_key FROM datasets WHERE name = %s LIMIT 1", (dataset_name,))
    res = cursor.fetchone()
    cursor.close()
    if res is not None:
        s3_key = res[0]
        s3_url = get_s3_url(s3_key)
        return render_template('dataset.html', download_url=s3_url)
    return "Dataset does not exist", 404

