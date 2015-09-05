from flask import request, render_template, redirect, url_for

from app.utils.s3 import s3_upload, get_s3_url

ALLOWED_EXTENSIONS = ['csv', 'json']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def new():
    dataset = request.files['dataset']
    if dataset and allowed_file(dataset.filename):
        filename = s3_upload(dataset)
        return redirect(url_for('datasets_view', dataset_name=filename))
    return redirect(url_for('index'))

def view(dataset_name):
    s3_url = get_s3_url(dataset_name)
    if not s3_url:
        return "Dataset does not exist", 404
    return render_template('dataset.html', download_url=s3_url)

