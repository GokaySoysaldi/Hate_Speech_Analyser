from tkinter import Variable
import nltk, re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import request
from flask import jsonify
from flask import Flask, render_template
import warnings
warnings.filterwarnings("ignore")


def preprocess_tweet(tweet):
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    return result



def make_prediction(tweet):
    model = pickle.load(open("pickles/lrrr2", "rb"))
    processed = preprocess_tweet(tweet)
    lst = []
    lst.append(processed)
    vec = pickle.load(open("pickles/vect", "rb"))
    vectorized = vec.transform(lst)
    pred = model.predict(vectorized)
    mapping = {0: 'Non-Hate Speech', 1: 'Hate Speech'}
    prediction = mapping[pred[0]]
    return prediction



app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    if(text):
        pred_class = make_prediction(text)

        return render_template('index.html',
                                variable=pred_class)
                               