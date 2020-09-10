from flask import Blueprint, render_template, url_for
from werkzeug.utils import redirect
from true_review.models import Movies, Reviews
from .. import db
from werkzeug.utils import redirect
import requests
from true_review.forms import CommentForm

bp = Blueprint('movies', __name__, url_prefix='/movies')


@bp.route('/movie/<int:movie_code>/<int:isFirstRender>/')
def movie(movie_code: int, isFirstRender: int):
    """
    rendering movie_detail page which is selected in movie_list page
    :param  movie_code:     movie identify code from naver movie page
            isFirstRender:  0 if comment is not saved or 1
    """
    movie = db.session.query(Movies).filter_by(code=movie_code).first()
    if not movie:
        return redirect(url_for('movies._list'))
    review_list = Reviews.query.filter_by(
        movie_id=movie.id).order_by(Reviews.text_rank.desc())
    form = CommentForm()
    return render_template('movies/movie_detail.html', movie=movie, form=form, isFirstRender=isFirstRender, review_list=review_list, comment=None)


@bp.route('/list/')
def _list():
    """
    rendering movie_list page
    """
    movie_list = Movies.query.order_by(Movies.create_date.desc())
    return render_template('movies/movie_list.html', movie_list=movie_list)
