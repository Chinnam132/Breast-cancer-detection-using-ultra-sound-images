import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
 
 
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from pathlib import Path
from PIL import Image
import os
from displayaug import *
import cv2
import pandas as pd

import os
app = Flask(__name__)
app.secret_key='detection'
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
 


 


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/registera")
def registera():
    return render_template("register.html")

@app.route("/logina")
def logina():
    return render_template("login.html")

@app.route("/predicta")
def predicta():
    return render_template("predict.html")


@app.route("/predictionoutputa")
def predictoutputa():
    return render_template("predictionoutput.html")


@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/menu.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")
             
app.static_folder = 'static'

    
# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route('/logouta')
def logout():
    # Clear the session data
    session.clear()
    return redirect(url_for('logina'))
    
def generate_recommendation(condition):
    data = ""
    print(condition)
    if condition == "Hemorrhagic":
        data = "Brain Hemorrhagic Detected in CT Scan"
    elif condition == "NORMAL":
        data = "CT Scan Image is Normal"
      
    return data


@app.route('/prediction1', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        classes = {0:"benignstage1",1:"benignstage2",2:"benignstage3",3:"malignantstage1",4:"malignantstage2",5:"malignantstage3",6:"normal"}
        
        # Get the image file from the request
        model = load_model('breastcancer.h5')
        image_file = request.files['image']
        print("Received image file:", image_file)
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        #image = Image.open(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict using the ensemble model
        result = np.argmax(model.predict(img_array))
        prediction = classes[result]
        print(prediction)
        predicted_class = prediction
        path="static/img/" 
        prediction = classes[result]
        print(prediction)
        
        DT = object()
        DT = DisplayAug()
        DT.readImage(file_path)
        DT.removeNoise()
        DT.displayAug()

        return render_template('predictionoutput.html', au='augment.png',prediction={'p':predicted_class,'image':os.path.join(app.config['UPLOAD_FOLDER'], filename),'data':predicted_class})
if __name__ == "__main__":
    app.run(debug=True)

     
     