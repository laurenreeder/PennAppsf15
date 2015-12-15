from flask import Flask, render_template, g
from keys import s3_access_key, s3_secret_key
from routes import register_routes
import logging

app = Flask(__name__)

app.config['S3_KEY'] = s3_access_key
app.config['S3_SECRET'] = s3_secret_key
app.config['S3_BUCKET'] = 'wissmann-reeder'
app.config['UPLOAD_FOLDER'] = './uploads'
@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.ERROR)

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
app.debug = True

register_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
