from flask import Flask
import os

def create_app():
    # Tell Flask where the templates folder is
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates'))

    from .routes import main
    app.register_blueprint(main)

    return app
