# -*- coding: utf-8 -*-
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from se489_group_project import logger
from se489_group_project.pipeline.pipeline_prediction import PredictionPipeline
from se489_group_project.utility.common import decodeImage

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    """
    ClientApp class to init the prediction pipeline.

    Attributes
    ----------
    filename : str
        The filename of the input image.
    classifier : PredictionPipeline
        The prediction pipeline object.
    """

    def __init__(self):
        """
        Initialize the ClientApp class with the filename and the prediction pipeline object.
        """
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    """
    Renders the index.html homepage.

    Returns
    -------
    str
        The rendered index.html for the home page.

    """
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    """
    Handles the training route.

    Returns
    -------
    str
        The response that the training is done successfully.

    """

    logger.info("running main.py")
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    """
    Handles the prediction route.

    Receives the image from the request, decodes it, and predicts the result using the prediction pipeline.

    Returns
    -------
    flask.Response
       JSON respons that contains the result of the prediction.

    """
    image = request.json["image"]
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)  # for AWS
