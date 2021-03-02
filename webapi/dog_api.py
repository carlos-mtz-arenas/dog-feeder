from flask import request
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

from uuid import uuid4 as uuid

import os

from localml import test_utils


(model, classes) = test_utils.get_model()
IMAGE_PREDICTION_DIR = os.path.abspath(os.environ['API_PREDICT_IMAGES'])


class DogAPI(Resource):

    def get(self):
        """
        Gets all the available classes supported by the model.
        """
        return {"classes": classes}

    def post(self):
        file = request.files['picture']
        extension = os.path.splitext(file.filename)[1]
        file_name = str(uuid()) + extension
        full_path = os.path.join(IMAGE_PREDICTION_DIR, file_name)
        file.save(full_path)
        prediction = self.predict(full_path)
        return {"prediction": prediction['class']}

    def predict(self, full_path):
        print('Loaded model uses classes [{}]'.format(classes))
        pred = test_utils.predict_class_for_image(model, classes, full_path)
        print('Pred=[{}]; Class=[{}]'.format(
            pred['prediction'], pred['class']))
        return pred
