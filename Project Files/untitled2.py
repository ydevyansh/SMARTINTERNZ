#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        model = joblib.load("model.pkl")
        
        # Get values through input bars
        n = request.form.get("n")
        p = request.form.get("p")
        k = request.form.get("k")
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")
        ph = request.form.get("ph")
        rainfall = request.form.get("rainfall")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]]
                         , columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        
        # Get prediction
        prediction = model.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
    
