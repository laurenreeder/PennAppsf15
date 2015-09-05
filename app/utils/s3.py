from uuid import uuid4
import boto
import os.path
from flask import current_app as app
from werkzeug import secure_filename

def s3_upload(source_file,acl='public-read'):
    source_filename = secure_filename(source_file.filename)
    source_extension = os.path.splitext(source_filename)[1]

    destination_filename = uuid4().hex + source_extension

    # Connect to S3
    conn = boto.connect_s3(app.config["S3_KEY"], app.config["S3_SECRET"])
    b = conn.get_bucket(app.config["S3_BUCKET"])

    # Upload the File
    print destination_filename
    sml = b.new_key(destination_filename)
    sml.set_contents_from_string(source_file.read())

    # Set the file's permissions.
    sml.set_acl(acl)

    return destination_filename

def get_s3_url(filename):
    # Connect to S3
    conn = boto.connect_s3(app.config["S3_KEY"], app.config["S3_SECRET"])
    b = conn.get_bucket(app.config["S3_BUCKET"])
    key = b.get_key(filename)
    if key:
        return key.generate_url(3600)
    else:
        return None


