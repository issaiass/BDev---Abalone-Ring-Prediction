# Abalone Ring Prediction  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/issaiass/BDev---Abalone-Ring-Prediction/master/streamlit.py)


## Description of the repository

- A jupyter notebook.
- The Abalone Dataset found [here](https://archive.ics.uci.edu/ml/datasets/Abalone).
- The final model file and jupyter notebooks

## Dataset Understanding

The task is to predict the number of rights of the abalone.  An abalone is a kind of sea animal with a shell and the age is predicted doing several measurements by microscope.  The original data was not used by Machine Learning algorithms and was for investigation purposes.

In the end i do simple investigations like:
- What kind of relationships is related to the data?  
- Histogram distribution of the data
- Which is the relationship between sex of the animal and rings?

## Data Understanding (Access and Explore)

Exploration of the dataset is done when we perform different queries that only are simple insights we will focus more on the process of making a model, load and consume in an app.

## Data Preparation (Cleaning)

The dataset is pre-prepared and it is no necessary to have cleaning
It consists of 4177 number of instances and 8 number of attributes measured in the next table

|Name|Data|Type	Meas.|Description|
|----|----|----|----|
|Sex|nominal||M F and I (infant)|
|Length|continuous|mm|Longest shell measurement
|Diameter|continuous|mm|perpendicular to length
|Height|continuous|mm|with meat in shell
|Whole weight|continuous|grams|whole abalone
|Shucked weight|continuous|grams|weight of meat
|Viscera weight|continuous|grams|gut weight (after bleeding)
|Shell weight|continuous|grams|after being dried|
|Rings|integer||+1.5 gives the age in years|

## Modelling

We used 3 models specifically only for machine learning.  A support vector regressor, a knn regressor and a random forest regressor, please check the results section for understanding of the results and possible improvemnents

### Evaluation

The model evaluation was done and the MSE has been used as a metric and also with the scikit-learn cross validation with a fold of 5.

### Deployment

We used streamlit to share the app so you can see how to serve a machine learning model using this incredible tool for fast deployment.

## Results

Severals models where used concluding that he Suport Vector Regressor was the one with a best metric MSE = 1.509.  We also explored other machine learning algorithms like Random Forest Regresion and KNN Regression with not best results as the previous one and recommends to use an ensemble method for get better metrics using XGBoost in example.

## Running the application

- Open your conda environment
- Do an install of the requirements `pip install -r requirements.txt`
- Run the streamlit app by `streamlit run streamlit.py`
- Play and have fun