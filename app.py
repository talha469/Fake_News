import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense,Dropout
from keras.layers import Dense,LSTM
from keras.models import load_model
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from pmdarima import auto_arima
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)
model = load_model('Fake_News.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    messages = request.form['text']
    corpus = []
    voc_size = 5000
    for i in range(1):
        review = re.sub('[^a-zA-Z]', ' ', messages)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)

        corpus.append(review)

    onehot = [one_hot(words, voc_size) for words in corpus]
    sentence_length = 40
    embeded = pad_sequences(onehot, padding='pre', maxlen=sentence_length)
    x_final = np.array(embeded)
    prediction = model.predict_classes(x_final)


    output = prediction




    print(output)
    if output[0][0] == 0:
        output = 'News is Trusted'
    elif output[0][0] ==1 :
        output = 'Fake News'

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)