from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image




from PIL import Image
# import numpy as np
from flask import Flask, request,render_template,redirect, url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app=Flask(__name__,static_url_path='/static')

MODEL_PATH = 'final-asl.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')     


def model_predict(image, model):
    img = Image.open(image)
    img = img.resize((64, 64))

    x = np.array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index2.html')

@app.route('/upload',methods=['GET'])
def uploadPage():
    return render_template('image_upload.html')

@app.route('/predict',methods=['POST'])
def predict():
    # if request.method=='POST':
    imageFile = request.files['imageFile']
    # imagePath = "./images/"+imageFile.filename
    # imageFile.save(imagePath)
    preds = model_predict(imageFile, model)
    top_prediction = np.argmax(preds)
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    result = class_labels[top_prediction]              # Convert to string
    return render_template('image_upload.html', result=result)

# return None

    # return render_template('index.html')


if __name__ =='__main__':
    app.run(port=3000,debug=True)