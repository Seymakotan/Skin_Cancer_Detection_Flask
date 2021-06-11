# -*- coding: utf-8 -*-
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image as pil_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
# Keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app =Flask(__name__)


Model= load_model('models/model_SC_v2.h5')         

def getCalssName(classNo):
    if   classNo == 0: return 'Melanocytic nevi'
    elif classNo == 1: return 'Melanoma'
    elif classNo == 2: return 'Benign keratosis-like lesions '
    elif classNo == 3: return 'Basal cell carcinoma'
    elif classNo == 4: return 'Actinic keratoses'
    elif classNo == 5: return 'Vascular lesions'
    elif classNo == 6: return 'Dermatofibroma'


@app.route('/', methods=['GET'])
def hello():
    return render_template('index_v2.html')

@app.route('/',methods=['GET','POST'])
def predict():  
    imagefile=request.files['imagefile']
    image_path="./images/" +imagefile.filename
    imagefile.save(image_path)
    
    image=load_img(image_path,target_size=(120,90))
    
    image=img_to_array(image)
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image=preprocess_input(image)
    classIndex = Model.predict_classes(image)
    label=getCalssName(classIndex)
    classification=label
 
    return render_template('index_v2.html',prediction=classification)


if __name__=='__main__':
    app.run(port=3000,debug=True) 