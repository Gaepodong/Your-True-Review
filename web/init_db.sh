#! /bin/bash

rm -rf migrations
rm -rf true_review.db

flask db init
flask db migrate
flask db upgrade
