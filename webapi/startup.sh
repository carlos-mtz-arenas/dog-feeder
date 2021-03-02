#!/bin/bash

source ../venv/bin/activate

export API_PREDICT_IMAGES="./prediction_images"
export API_USER="fake-user"
export API_PWD="pwdfake"

python main.py