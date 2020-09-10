from bs4 import BeautifulSoup
import requests
import csv
import os
from datetime import datetime
from true_review import db
from true_review.models import Movies, Reviews


def get_title_and_score(code):
    """
    get movie title from naver movie page with movie_code
    :param code: int movie_code
    :return title: string movie_title
    """
    url = 'https://movie.naver.com/movie/bi/mi/basic.nhn?code=' + str(code)
    response = requests.get(url.format(1))
    if response.status_code == 500:
        print("Movie code error")
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h3', class_='h_movie').find('a').text
    score = ''.join([i.text for i in soup.find('div', class_='score score_left').find('div', class_='star_score').find_all('em')])
    return title, float(score)


def get_image_url(code):
    """
    get movie_image url from naver movie page with movie_code
    :param code: int movie_code
    :return imageUrl['src']: string movie_image url
    """
    url = 'https://movie.naver.com/movie/bi/mi/photoViewPopup.nhn?movieCode=' + \
        str(code)
    response = requests.get(url.format(1))
    if response.status_code == 500:
        print("Movie code error")
    soup = BeautifulSoup(response.text, 'html.parser')
    imageUrl = soup.find('img')
    return imageUrl['src']


def get_already_registered_movie_codes():
    """
    get movie code list from DB
    :return movie_code_list: List already_existed_movie_codes
    """
    movie_code_list = db.session.query(Movies.code).distinct()
    return movie_code_list


def update_reviews(movie):
    """
    update  movie reviews.
    :param  movie: Movies data object
    :return Boolean: if there is no file, return false
    """
    path = 'ranked_reviews/' + str(movie.code) + '.csv'
    try:
        review_file = open(path, 'r', encoding='cp949')
    except FileNotFoundError as e:
        print("Error: ", e)
        return False

    rdr = csv.reader(review_file)
    try:
        for i, line in enumerate(rdr):
            if i == 0:
                continue
            if i == 1:
                # movie.review_score = line[4]
                movie.review_score = 8.94
            text_rank = float(line[1])
            content = line[2]
            pos_or_neg = int(line[3])
            reviews = Reviews(movie.id, text_rank, content, pos_or_neg)
            movie.review_set.append(reviews)
            db.session.add(movie)
    except Exception as e:
        print("Error: ", e)
        print("Error movie code: {}".format(movie.code))
    review_file.close()
    return True


def update_movies_and_reviews():
    """
    update movies and reviews without duplication
    """
    path_dir = 'ranked_reviews/'
    file_list = os.listdir(path_dir)
    movie_code_list = [int(i.split('.')[0]) for i in file_list]
    get_already_registered_movie_code = [
        i[0] for i in list(get_already_registered_movie_codes())]

    for movie_code in get_already_registered_movie_code:
        try:
            movie_code_list.remove(movie_code)
        except:
            pass

    for movie_code in movie_code_list:
        title, score = get_title_and_score(movie_code)
        imageUrl = get_image_url(movie_code)
        movie = Movies(title, movie_code, datetime.now(), imageUrl, score)
        if update_reviews(movie) is False:
            return
    try:
        db.session.commit()
    except:
        return
