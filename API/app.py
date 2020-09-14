from flask import Flask, request
import torch
import torch.nn as nn
import get_model as m


app = Flask(__name__)

test = m
model = m.get_model()
soft = nn.Softmax(dim=1)


@app.route("/")
def hello():
    return "Hello goorm!"


@app.route("/predict", methods=['POST'])
def predict():
    req = request.values.to_dict()
    sentence = req['text']
    # sentence = "언젠가 다시 보고 싶은 영화"
    print(req)

    rating, posneg = model.inference(sentence)
    rating_soft = soft(rating)
    posneg_soft = soft(posneg)

    ret = dict()
    ret['emotion_percent'] = round(100 - (torch.max(rating_soft).item() * 100), 2)
    ret['movie_rating'] = torch.argmax(rating_soft).item()
    ret['pos_or_neg'] = torch.argmax(posneg_soft).item()  # 0:부정, 1:긍정
    print(ret)

    return ret


if __name__ == "__main__":
    app.run(host='0.0.0.0')

