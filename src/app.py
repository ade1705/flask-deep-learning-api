from fastai import *
from fastai.vision import *
import fastai
import yaml
import sys
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from flask import Flask, flash, request, redirect, url_for, session, jsonify
import requests
import torch
import json

from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin


with open("src/config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)

app = Flask(__name__)
export_file_name = 'export.pkl'
UPLOAD_FOLDER = '/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(path=".", model_name="model.pkl"):
    learn = load_learner(path, export_file_name)
    return learn


def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img


def load_image_bytes(raw_bytes: ByteString) -> Image:
    img = open_image(BytesIO(raw_bytes))
    return img


def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:
    pred_class, pred_idx, outputs = model.predict(img)
    pred_probs = outputs / sum(outputs)
    pred_probs = pred_probs.tolist()
    predictions = []
    for image_class, output, prob in zip(model.data.classes, outputs.tolist(), pred_probs):
        output = round(output, 1)
        prob = round(prob, 2)
        predictions.append(
            {"class": image_class.replace("_", " "), "output": output, "prob": prob}
        )

    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0:n]
    return {"class": str(pred_class), "predictions": predictions}

def classify_file(url: str):
    img = open_image(url)
    res = predict(img)
    return res


@app.route('/file-upload', methods=['POST'])
def upload_image_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : classify_file(app.config['UPLOAD_FOLDER'] + filename)})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = load_image_url(url)
    else:
        bytes = flask.request.files['file'].read()
        img = load_image_bytes(bytes)
    res = predict(img)
    return flask.jsonify(res)


@app.route('/api/classes', methods=['GET'])
def classes():
    classes = sorted(model.data.classes)
    return flask.jsonify(classes)


# @app.route('/ping', methods=['GET'])
# def ping():
#     return "pong"

@app.route('/ping', methods=['GET', 'POST'])
def ping():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response="Whatever you wish too return"
    return response

    

@app.route('/api/upload', methods=['GET'])
def classify():
    return "classify"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_request():
    app.jinja_env.cache = {}


model = load_model('models')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)

    if "prepare" not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=True, host='0.0.0.0', port=port)
        # app.run(host='0.0.0.0', port=port)
