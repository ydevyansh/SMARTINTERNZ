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
        model2 = joblib.load("model2.pkl")
        model3 = joblib.load("model3.pkl")
        le = joblib.load("le.pkl")
        
        # Get values through input bars
        n = request.form["n"]
        p = request.form["p"]
        k = request.form["k"]
        temperature = request.form["temperature"]
        humidity = request.form["humidity"]
        ph = request.form["ph"]
        rainfall = request.form["rainfall"]
        
        # Put inputs to dataframe
        X = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        
        # Get prediction
        prediction = model.predict(X)[0]
        prediction = le.inverse_transform(prediction)[0]
        print(prediction)
        
        dict = {prediction: 1}
        prediction = model2.predict(X)[0]
        prediction = le.inverse_transform(prediction)[0]
        print(prediction)
        
        if prediction in dict:
            dict[prediction] = dict[prediction] + 1
        else:
            dict[prediction] = 1
            
        prediction = model3.predict(X)[0]
        prediction = le.inverse_transform(prediction)[0]
        print(prediction)
        
        if prediction in dict:
            dict[prediction] = dict[prediction] + 1
        else:
            dict[prediction] = 1
        
        val = dict[prediction]
        for x in dict :
            if(dict[x] > val):
                val = dict[x]
                prediction = x
                
        prediction = "Best crop for the soil is " + prediction
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True, port=3001)
    
