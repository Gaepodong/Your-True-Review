from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import config

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    """
    start true_review app
    """
    app = Flask(__name__)

    app.config.from_object(config)

    db.init_app(app)
    migrate.init_app(app, db)
    from . import models

    from .views import main_views, movies_views, comment_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(movies_views.bp)
    app.register_blueprint(comment_views.bp)

    return app
