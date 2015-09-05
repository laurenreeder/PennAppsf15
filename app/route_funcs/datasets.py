from flask import request, render_template

from app.utils.s3 import s3_upload

ALLOWED_EXTENSIONS = ['csv', 'json']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def new():
    dataset = request.files['dataset']
    if dataset and allowed_file(dataset.filename):
        filename = s3_upload(dataset)
        return "Success"
    return "Fail"

def view(dataset_name):
    return render_template('dataset.html', dataset_name=dataset_name)

