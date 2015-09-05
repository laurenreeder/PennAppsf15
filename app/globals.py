import psycopg2
from flask import g

DATABASE = 'postgres://localhost/metadata'

def connect_to_database():
    return psycopg2.connect(DATABASE)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_to_database()
    return db

