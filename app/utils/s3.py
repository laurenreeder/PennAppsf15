from uuid import uuid4
import boto
import os
from flask import current_app as app
from werkzeug import secure_filename
import urllib2

def s3_upload(source_file,acl='public-read'):
    curr_file = open(source_file)
    source_filename = secure_filename(source_file)
    source_extension = file_extension(source_filename)
    destination_filename = uuid4().hex + source_extension

    # Connect to S3
    conn = boto.connect_s3(app.config["S3_KEY"], app.config["S3_SECRET"])
    b = conn.get_bucket(app.config["S3_BUCKET"])

    # Upload the File
    print destination_filename
    sml = b.new_key(destination_filename)
    sml.set_contents_from_string(curr_file.read())

    # Set the file's permissions.
    sml.set_acl(acl)

    return destination_filename

def file_extension(filepath):
    return filepath.split('?')[0].split('/')[-1].split('.')[-1]

def s3_download(source_url,acl='public-read'):

    source_filename = source_url.split('?')[0].split('/')[-1]
    source_extension = file_extension(source_filename)

    destination_filename = uuid4().hex + source_extension

    # Connect to S3
    conn = boto.connect_s3(app.config["S3_KEY"], app.config["S3_SECRET"])
    b = conn.get_bucket(app.config["S3_BUCKET"])

    # Upload the File
    print destination_filename
    sml = b.new_key(destination_filename)
    source_file = urllib2.urlopen(source_url)

    with open('./datasets/%s' % destination_filename, 'w') as f:
        f.write(source_file.read())

    with open('./datasets/%s' % destination_filename, 'r') as f:
        sml.set_contents_from_file(f)

    os.remove('./datasets/%s' % destination_filename)

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


