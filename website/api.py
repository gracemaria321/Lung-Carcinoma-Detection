# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:17:55 2021

@author: malilanz
"""

import os

from flask import Flask
from flask import request
from flask import render_template

import numpy as np

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


UPLOAD_FOLDER = 'E:/Project/Project/website/test_images'

MODEL_PATH = 'E:/Project/Project/website/models/new_model1.h5'

app = Flask(__name__)

def predict_val(path):
    res = []
    img_path = path
    classes = ["Benign", "Malignant", "Normal"]
    model = load_model(MODEL_PATH)
    
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    img_data = preprocess_input(x)
    a = np.argmax(model.predict(img_data))
    val = classes[a.item(0)]
    res.append(val)
    
    return res[0]


@app.route("/",methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:  
            image_location = os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
            pred = predict_val(image_location)
            if(str(pred) == "Malignant"):
                res1 = "malignant"
                res2 = "Immediate treatment can cure faster.."
            elif(str(pred) == "Benign"):
                res1 = "benign"
                res2 = "Take checkups regularly. Be safe.."
            else:
                res1 = "normal"
                res2 = "No issues found. Enjoy your day.."
                
            return render_template("index.html", display="inline-block", result = res1, remark = res2)
    return render_template("index.html", display="none", result = "", remark = "")


if __name__ == '__main__':
    
    app.run()
