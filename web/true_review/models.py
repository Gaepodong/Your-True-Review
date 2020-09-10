from true_review import db


class Movies(db.Model):
    __tablename__ = 'movies'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    code = db.Column(db.Integer, nullable=True, unique=True)
    create_date = db.Column(db.DateTime(), nullable=False)
    image_path = db.Column(db.Text(), nullable=True)
    score = db.Column(db.Float, nullable=True)
    review_score = db.Column(db.Float, nullable=True)
    db.UniqueConstraint('id', 'code')

    reviews = db.relationship(
        "Reviews", backref=db.backref('movie', order_by=id))

    def __init__(self, title, code, create_date, image_path, score):
        self.title = title
        self.code = code
        self.create_date = create_date
        self.image_path = image_path
        self.score = score


class Reviews(db.Model):
    __tablename__ = 'reviews'

    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey(
        'movies.id', ondelete='CASCADE'))
    text_rank = db.Column(db.Float, nullable=True)
    content = db.Column(db.Text(), nullable=False)
    pos_or_neg = db.Column(db.Boolean, nullable=False)
    reviews = db.relationship('Movies', backref=db.backref('review_set'))

    def __init__(self, movie_id, text_rank, content, pos_or_neg):
        self.movie_id = movie_id
        self.text_rank = text_rank
        self.content = content
        self.pos_or_neg = pos_or_neg


class Comments(db.Model):
    __tablename__ = 'comments'

    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey(
        'movies.id', ondelete='CASCADE'))
    content = db.Column(db.Text(), nullable=False)
    movie_rating = db.Column(db.Integer, nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    emotion_percent = db.Column(db.Float, nullable=True)
    pos_or_neg = db.Column(db.Boolean, nullable=True)
    movie = db.relationship('Movies', backref=db.backref('comment_set'))

    def __init__(self, movie_id, content, movie_rating, create_date, emotion_percent, pos_or_neg):
        self.movie_id = movie_id
        self.content = content
        self.movie_rating = movie_rating
        self.create_date = create_date
        self.emotion_percent = emotion_percent
        self.pos_or_neg = pos_or_neg
