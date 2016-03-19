import psycopg2
from flask import g
from keys import rds_password

def remote_connect():
    return psycopg2.connect(host='image-classification.cfgdweprellz.us-east-1.rds.amazonaws.com', port=5432,
                            user='leap', password=rds_password, database='images')

def local_connect():
    return psycopg2.connect(host='localhost', port=5432, database='images')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = local_connect()
    return db

