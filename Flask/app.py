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
with open('../models/GBCmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/GBCmodelScaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

client = EventAPIClient1()
@app.route('/')
@app.route('/home')
def home():
    
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")



@app.route('/predict')
def predict():
    
    row = client.collect()
    data =pd.DataFrame(row)
    
    
    X = get_example_X_y(data, scaler)


    pred =model.predict_proba(X)
    return render_template('results.html', pred=np.around(pred[0][1], decimals=4), name=data['name'][0] )
    




if __name__=="__main__":

    
    
    app.run(debug=True)