import flask

from flask import request, jsonify

import sys
import os
from uuid import uuid4 as uuid

from localml import test_utils

app = flask.Flask("dog-feeder-api")

# app configuration ::::::::::::::::::::::::::::::::::: BEGINS
# log level for debugging purposes
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = os.path.abspath('./prediction_images')
# app configuration ::::::::::::::::::::::::::::::::::: ENDS

(model, classes) = test_utils.get_model()


def predict(full_path):
    print('Loaded model uses classes [{}]'.format(classes))
    pred = test_utils.predict_class_for_image(model, classes, full_path)
    print('Pred=[{}]; Class=[{}]'.format(pred['prediction'], pred['class']))
    return pred


@app.route('/', methods=['GET'])
def home():
    print(request)
    return jsonify({"message": "yay"})


@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['picture']
    extension = os.path.splitext(file.filename)[1]
    file_name = str(uuid()) + extension
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_path)
    prediction = predict(full_path)
    return jsonify({"prediction": prediction['class']})


@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Gets all the available classes supported by the model.
    """
    return jsonify({"classes": classes})


@app.errorhandler(Exception)
def handle_500(e):
    """
    Handles all the errors from the server and
    applys a JSON format for the response
    """
    return jsonify({"error": True, "message": str(e)}), 500


app.run()
