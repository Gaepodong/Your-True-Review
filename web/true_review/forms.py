from flask_wtf import FlaskForm
from wtforms import StringField, TextField
from wtforms.validators import DataRequired, Length


class CommentForm(FlaskForm):
    """
    comment form which has validator
    """
    content = TextField('내용', validators=[Length(
        min=20, max=330, message="Comment must be between %(min)d and %(max)d characters")])
