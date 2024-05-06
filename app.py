import pickle
import os
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##Load the model
tokenized_test_data=pickle.load(open('tokenized_test_data.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
   data=request.json['data']
   prediction = tokenized_test_data.predict(data)
   return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
   
   