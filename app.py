# -*- coding: utf-8 -*-
import os

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from se489_group_project import logger
from se489_group_project.pipeline.pipeline_prediction import PredictionPipeline
from se489_group_project.utility.common import decodeImage

# from flask import Flask, jsonify, render_template, request
# from flask_cors import CORS, cross_origin


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

# app = Flask(__name__)
# CORS(app)

# refactor the code to use FastAPI instead of Flask
app = FastAPI()

# Initialize Jinja2 template engine
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now, allow all origins, not for production settings
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


class ImageRequest(BaseModel):
    image: str


# Dependency injection function
def get_client_app():
    return ClientApp()


# @app.route("/", methods=["GET"])
# @cross_origin()
@app.get("/")
def home(request: Request):
    """
    Renders the index.html homepage.

    Returns
    -------
    str
        The rendered index.html for the home page.

    """
    # return render_template("index.html")
    # Utilize FastAPI's Jinja2Templates to render the index.html
    return templates.TemplateResponse("index.html", {"request": request})


# @app.route("/train", methods=["GET", "POST"])
# @cross_origin()
@app.get("/train")
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


# added to handle POST requests for the training route
@app.post("/train")
def post_train_route():
    """
    Handles POST requests for the training route.
    """
    logger.info("running main.py")
    os.system("python main.py")
    return {"message": "Training done successfully with POST!"}


# @app.route("/predict", methods=["POST"])
# @cross_origin()
@app.post("/predict")
def predictRoute(request: ImageRequest, clApp: ClientApp = Depends(get_client_app)):
    """
    Handles the prediction route.

    Receives the image from the request, decodes it, and predicts the result using the prediction pipeline.

    Returns
    -------
    flask.Response
       JSON respons that contains the result of the prediction.

    """
    # image = request.json["image"]
    # decodeImage(image, clApp.filename)
    # #return jsonify(result)

    decodeImage(request.image, clApp.filename)
    result = clApp.classifier.predict()
    return JSONResponse(content=result)


if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host="0.0.0.0", port=8080)  # for AWS
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

# To run the app, execute the following command:
# uvicorn app:app --reload --port 8080
