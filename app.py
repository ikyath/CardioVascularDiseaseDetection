#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:40:15 2020

@author: ikyathvarmadantuluri
"""

import numpy as np
import pickle 
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction
    
    if output==1:
        prediction = "More prone to disease"
    else:
        prediction = "Less prone to disease"
        
    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)

    
