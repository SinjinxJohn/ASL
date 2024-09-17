from __future__ import division, print_function
from datetime import datetime
import tempfile
import pyrebase
import sys
import os
import glob
import re
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import requests
from gtts import gTTS
from io import BytesIO

config = {
    "apiKey": "AIzaSyBV-wsWGwWorvM10tCl8MX4_gjq5uHC8Q8",
    "authDomain": "asl-project-bca0c.firebaseapp.com",
    "projectId": "asl-project-bca0c",
    "storageBucket": "asl-project-bca0c.appspot.com",
    "messagingSenderId": "272789461101",
    "appId": "1:272789461101:web:43819fdf4968e2f9814764",
    "databaseURL": None
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

app = Flask(__name__, static_url_path='/static')

MODEL_PATH = 'final-asl.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(image, model):
    img = image.resize((64, 64))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index2.html')

@app.route('/upload', methods=['GET'])
def uploadPage():
    return render_template('image_upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'imageFile' not in request.files:
            return "No file part"
        imageFile = request.files['imageFile']
        if imageFile.filename == '':
            return "No selected file"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
            imageFile.save(temp_file_path)

        storage.child("images/" + imageFile.filename).put(temp_file_path)
        download_url = storage.child("images/" + imageFile.filename).get_url(None)

        # Download the image and open it with PIL
        response = requests.get(download_url)
        img = Image.open(BytesIO(response.content))

        preds = model_predict(img, model)
        top_prediction = np.argmax(preds)
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        result = class_labels[top_prediction]

        os.remove(temp_file_path)

        asr_to_speech_url = 'http://127.0.0.1:3000/asr_to_speech'
        response = requests.post(asr_to_speech_url,json={'recognized_text':result})

        if response.status_code == 200:
            audio_file = response.json().get('audio_file')
            return render_template('image_upload.html', result=result, audio_file=audio_file)
        else:
            return "Error in text-to-speech conversion", response.status_code
        # return render_template('image_upload.html', result=result)
    except Exception as e:
        print(e)
        return str(e)

@app.route('/asr_to_speech', methods=['POST'])
def asr_to_speech():
    try:
        data = request.json
        recognized_text = data.get('recognized_text', '')  # Text from ASL recognition model
        
        if not recognized_text:
            return jsonify({'message': 'No recognized text provided'}), 400
        
        tts_message = f"The predicted word or letter is {recognized_text}"
        # Generate speech from the recognized text
        tts = gTTS(text=tts_message, lang='en')
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_file_name = f'output_{timestamp}.mp3'
        audio_file_path = os.path.join('static', audio_file_name)
        tts.save(audio_file_path)
        
        return jsonify({'message': 'Speech generated', 'audio_file': audio_file_name}), 200
    
    except Exception as e:
        print(e)
        return str(e), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)
