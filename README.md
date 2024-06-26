# se489_group_project

# Kidney Disease diagnosis DL model

## Project Description

The project aims to develop a deep learning model using the [1] VGG16 architecture to detect kidney tumors in CT scans. The model will be trained on the [2] CT Kidney Dataset, which contains images of normal kidneys and kidneys with tumors. By leveraging the VGG16 model's ability to extract intricate features from images, the goal is to create a reliable tool for early detection of kidney tumors, aiding in timely medical interventions and improved patient outcomes.

### Below highlights the technology stack:

Cookiecutter template – file structure.

GitHub – Version control.

DVC – data version control.

MLflow – compare trained models

Docker – load models into containers

AWS – host models inside of a web interface

Keras – User-friendly deep learning model development

Jupyter Notebook – Interactive development and experimentation

TensorFlow –Loading/Training models, Data preprocessing, Data Generator Creation

Pre-Commit - Runs automated code quality checks before each commit.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── se489_group_project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

### Create a conda environment after opening the repository

```bash
conda create -n grp_env python=3.8 -y
```

```bash
conda activate
```

```bash
pip install -r requirements.txt
```

```bash
# run following command
uvicorn app:app --reload --port 8080
```

Now,
```bash
open localhost and port(8080) / http://127.0.0.1:8080
```
To trigger training visit http://127.0.0.1:8080/train

##### cmd (to check your model on mlflow ui without logging to dagshub/mlflow remote)
- cd se489_group_project
- cd components
- mlflow ui

## Please update files in below order

1. Update config.yaml
2. Update params.yaml
3. Update the model_classes
4. Update the configurationManager in se489_group_project config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the dvc.yaml <optional>
9. app.py

### dagshub

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI= https://dagshub.com/RobinMacwan/group_project_se489.mlflow

export MLFLOW_TRACKING_USERNAME=RobinMacwan

export MLFLOW_TRACKING_PASSWORD= <available upon request>

### LINK TO DAGSHUB EXPERIMENTS
https://dagshub.com/eTroupe5201/GroupProjectSE489/experiments/#/

### LINK TO MLFLOW REPORTS
https://dagshub.com/eTroupe5201/GroupProjectSE489.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

```

### DVC cmd for looking up code dependancy

1. dvc init
2. dvc dag

- for example

![Dependancy diagram](image.png)

## Setting Up Pre-Commit Hooks

Run as Adminstrator in Anaconda.Navigator to ensure proper installation.
Detailed instruction in .md file.

```bash
pip install -r requirements.txt
```
After running requirements.txt run the following command in the root directory.
```bash
pre-commit install
```.
This command installs pre-commit hooks. Pre-commit hooks will run automatically on every commit

If you want to remove pre-commit hooks and disable automatic execution. Run the following command
```bash
pre-commit uninstall
```
To clear the cache. Run the following command
```bash
pre-commit clean
To run pre-commit hooks on all files in the repository manually.
```bash
pre-commit run --all-files
```


To run black
```bash
black .
```

To run isort
```bash
isort .-v
```

To run interrogate
```bash
interrogate -v
```
## Setting Up Ruff
```bash
pip install -r requirements.txt
```
After running requirements.txt run the following command in the root directory.
```bash
ruff .

### REFERENCES/Citation

#[1] Model used : Vgg16 - https://keras.io/api/applications/vgg/

#[2] Test Data: CT Kidney dataset : https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/code

#[3] Code help: https://github.com/malleswarigelli/Chest_Disease_Image_Classification_

#[4] app.py/index.html: we use generic bootstrap ui
