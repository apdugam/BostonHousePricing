import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

### Start with Flash App (App will run from here!)
app = Flask(__name__)

## Load Pickle Files
reg_model = pickle.load(open("reg_model.pkl", "rb"))
scalar = pickle.load(open('scaling.pkl', 'rb'))

## Home Function will be executed when client requests it from URL
@app.route("/")
def home():
    return render_template('home.html')

## Create Predict API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    ## Get data from Json
    data = request.json['data']
    print(data)
    
    ## Standardize New Data
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    
    ## Now Pass the Std data to Model to Make Predictions
    output = reg_model.predict(new_data)
    print(output[0])
    
    ## Get output in jasonify
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    
    ## Get Data From WebApp
    data = [float(x) for x in request.form.values()]
    
    ## Standadizing the WebApp Data
    std_data = scalar.transform(np.array(list(data)).reshape(1, -1))
    print(std_data)
    
    ## Now Pass the Std data to Model to Make Predictions
    output = reg_model.predict(std_data)[0]
    
    return render_template("home.html", prediction_text=f"The Predicted Price of House is: {output}")
    
    
    
    

### Let's Run the App
if __name__ == "__main__":
    app.run(debug=True)