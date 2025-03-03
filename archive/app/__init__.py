from flask import Flask
import os

def create_app():
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    from .routes import routes
    app.register_blueprint(routes)
    
    return app