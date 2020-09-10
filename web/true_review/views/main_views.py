from flask import Blueprint, render_template, url_for
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import redirect
import config
from true_review.models import Movies
from true_review.update import update_movies_and_reviews


bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/update')
def update():
    """
    update movies and reviews from 'ranked_reviews' folder
    :return redirect to movie_.list
    """
    update_movies_and_reviews()
    return redirect(url_for('movies._list'))


@bp.route('/')
def index():
    """
    :return redirect to movies._list
    """
    return redirect(url_for('movies._list'))
