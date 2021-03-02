import flask

from flask import request, jsonify
from flask_restful import Api
from dog_api import DogAPI

import sys
import os
from uuid import uuid4 as uuid


app = flask.Flask("dog-feeder-api")

# app configuration ::::::::::::::::::::::::::::::::::: BEGINS
# log level for debugging purposes
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.environ['API_PREDICT_IMAGES'])
# app configuration ::::::::::::::::::::::::::::::::::: ENDS


@app.route('/', methods=['GET'])
def home():
    print(request)
    return jsonify({"message": "yay"})


@app.errorhandler(Exception)
def handle_500(e):
    """
    Handles all the errors from the server and
    applys a JSON format for the response
    """
    return jsonify({"error": True, "message": str(e)}), 500

# create the API
api = Api(app)

api.add_resource(DogAPI, '/dogs')

app.run()
