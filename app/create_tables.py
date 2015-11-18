import psycopg2
from app.globals import connect_to_database

conn = connect_to_database()
cursor = conn.cursor()
cursor.execute("CREATE TABLE datasets(name varchar primary key, s3_key varchar)")

cursor.close()

conn.commit()
conn.close()

