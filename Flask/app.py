from flask import Flask, render_template, url_for, request
import numpy as np
from sklearn.datasets import load_iris
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import random
from src.API import *
from src.gbc_predict import * 


app = Flask(__name__)
with open('models/GBCmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/GBCmodelScaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
@app.route('/')
@app.route('/home')
def home():
    
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")



@app.route('/predict')
def predict():
    client = EventAPIClient1()
    row = client.collect()
    data =pd.DataFrame(row)
    
    X = get_example_X_y(data, scaler)


    pred =model.predict(X)
    return render_template('results.html', data=pred)
    




if __name__=="__main__":

    
    
    app.run(debug=True)