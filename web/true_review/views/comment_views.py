from datetime import datetime
from flask import Blueprint, url_for, request, render_template
from werkzeug.utils import redirect
from true_review import db
from true_review.models import Comments, Movies, Reviews
from true_review.forms import CommentForm
import requests as req
import json
import os

bp = Blueprint('comment', __name__, url_prefix='/comment')


@bp.route('/get_and_analysis/<int:movie_id>', methods=('GET', 'POST'))
def get_and_analysis(movie_id):
    """
    get comment and analysis it to get its emotion(positive/negative) and star rating by model.
    :return rendering movie_detail page differently wheather or not the valid comment is saved.
    """
    movie = Movies.query.get_or_404(movie_id)
    review_list = Reviews.query.filter_by(
        movie_id=movie.id).order_by(Reviews.text_rank.desc())
    form = CommentForm()
    if request.method == 'POST' and form.validate_on_submit():
        response = req.post(os.environ.get("API_SERVER"),
                            data={'text': form.content.data}).json()
        emotion_percent = response['emotion_percent']
        movie_rating = response['movie_rating']
        pos_or_neg = response['pos_or_neg']

        comment = Comments(movie_id=movie_id, content=form.content.data,
                           movie_rating=movie_rating, create_date=datetime.now(), emotion_percent=emotion_percent, pos_or_neg=pos_or_neg)
        movie.comment_set.append(comment)
        db.session.commit()
        return render_template('movies/movie_detail.html', movie=movie, form=form, isFirstRender=0, review_list=review_list, comment=comment)
    return render_template('movies/movie_detail.html', movie=movie, form=form, isFirstRender=1, review_list=review_list, comment=None)
