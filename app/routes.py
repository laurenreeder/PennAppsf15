from flask import Blueprint, render_template

from route_funcs import datasets

def register_routes(app):

    @app.route("/")
    def index():
        return render_template('home.html')

    app.add_url_rule('/datasets/new', 'datasets_new', datasets.new, methods=['POST'])
    app.add_url_rule('/datasets/<dataset_name>', 'datasets_view', datasets.view, methods=['GET'])



