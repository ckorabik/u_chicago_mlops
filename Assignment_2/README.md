# ADSP 31021 Assignment 2
Chris Korabik

A good bit of this pipeline was created with assistance from Codex. I did a lot of manual work in data_selection.ipynb, and used Codex to help me expand the ingest and preprocessing steps into a full working pipeline.

## 1. Data selection and setup
Please see data_selection.ipynb for more details around the initial exploratory analysis, data preprocessing, and feature generation decisions.

## 2. MLOps Platform Selection and Configuration
I chose to go with MLFlow as my MLOps platform. This is primarily because I know it is one of the leading platforms utilized by companies, and I would like to get experience working with it. MLFlow is built to help with model versioning and monitoring, and it is a useful tool to help debug and develop models. MLFlow can also help me keep track of parameters, features, and metrics used in the model training process.

It is also free and open source.

## 3. E2E ML Pipeline Development
An end-to-end machine learning pipeline exists in `pipeline.py`.

All that needs to be run from a clean conda environment is:
- `pip install -r requirements.txt`
- `python pipeline.py`

## 4. Feature Store Integration
MLFlow's feature store is utilized in this project. We use sqlite as the feature store backend. These files are stored on my local machine, but for a more robust and collaborative option we could look to store the features in a remote database.

`pipeline.py` details the steps taken to store features locally.

## 5. Feature versioning
I wanted multiple feature sets to store, so I opted to create both versions in the same pipeline. I will be able to access either feature table later in the training session.

`pipeline.py` also details the feature versioning steps taken. We can toggle the feature version we use by setting `ACTIVE_FEATURE_VERSION`. Version 2 in this example is a smaller subset.

## 6. Experimentation and Model Training
For each of the two feature versions I defined, I attempted two different hyperparameter configurations. I chose to use XBoost as the ensemble regression model for each experiment.

The feature versions were changed by changing `ACTIVE_FEATURE_VERSION` as mentioned above, while model hyperparameters were altered in the XGBRegressor input within `training.py`.

Our pipeline utilizes MLFlow in a way that captures these differences with every run. 

See `docs/model-comparison.png` for proof of experiment tracking and comparison of the different runs.

## Steps to reproduce
This assumes you have access to the original dataset from Canvas and that it is stored in `/Assignment_2/data/`.

Navigate to the Assignment_2 folder: 
`cd Assignment_2`

Install dependencies: 
`pip install -r requirements.txt`

Run the pipeline: 
`python pipeline.py`

View MLFlow UI for model versioning and feature store information:
`mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000`

Then just visit `http://localhost:5000`!
