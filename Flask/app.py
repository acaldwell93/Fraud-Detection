from flask import Flask, render_template, url_for, request
import numpy as np
from sklearn.datasets import load_iris
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import random

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")



@app.route('/predict')
def predict():
    data1= request.form['a']


    arr = np.array([data1])

    pred =model.predict(arr)
    return render_template('results.html', data=pred)
    
    
    
    return render_template('predict.html')




if __name__=="__main__":
    app.run(debug=True)