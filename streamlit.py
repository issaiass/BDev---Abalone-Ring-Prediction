import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
import pickle

st.set_page_config(layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# load the model from disk
@st.cache
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

file_url = 'https://assets3.lottiefiles.com/packages/lf20_drzlffcc.json'
lottie_sea = load_lottieurl(file_url)
st_lottie(lottie_sea, speed=2, height=200, key="initial")


# Side Bar
abalone_df = pd.read_csv('abalone.csv')
min_vals = abalone_df.min()
max_vals = abalone_df.max()
st.sidebar.title('Abalone Predictive Data')
sex_box = st.sidebar.selectbox("Sex", ("Male", "Infant", "Female"))
lenght = st.sidebar.slider('Lenght', min_vals[1], max_vals[1], 0.01)
diameter = st.sidebar.slider('Diameter', min_vals[2], max_vals[2], 0.01)
height = st.sidebar.slider('Height', min_vals[3], max_vals[3], 0.01)
whole_weight = st.sidebar.slider('Whole Weight', min_vals[4], max_vals[4], 0.01)
shucked_weight = st.sidebar.slider('Shucked Weight', min_vals[5], max_vals[5], 0.01)
viscera_weight = st.sidebar.slider('Viscera Weight', min_vals[6], max_vals[6], 0.01)

# Midle Page
st.title('Abalone Age Prediction')
st.write('   Predicting the age of abalone from physical measurements.'
         'The age of abalone is determined by cutting the shell through'
         'the cone, staining it, and counting the number of rings through'
         'a microscope -- a boring and time-consuming task.  Other measurements, '
         'which are easier to obtain, are used to predict the age.  Further information, '
         'such as weather patterns and location (hence food availability)'
         'may be required to solve the problem.')

if sex_box == 'Male':
    sex_box = float(2)
if sex_box == 'Infant':
    sex_box = float(1)
if sex_box == 'Female':
    sex_box = float(0)

le = preprocessing.LabelEncoder()
le.fit(abalone_df['sex'])
abalone_df['sex'] = le.transform(abalone_df['sex'])


model = load_model()
arr = np.array([[sex_box, lenght, diameter, height, whole_weight, shucked_weight, viscera_weight]])

# predictions
preds = np.floor(model.predict(arr))
st.title('Your Prediction of Abalone Rings is ' + str(int(preds[0])))


# Data Columns
col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots()
    lenght = abalone_df.to_numpy()[:,1]
    dia = abalone_df.to_numpy()[:,2]
    ax.set_title("Relationship between Diameter and Lenght", size=20)
    ax.scatter(lenght, dia, c=dia, s=40, cmap=plt.cm.RdYlBu)
    ax.grid('on')
    ax.set_xlabel('lenght', size=15)
    ax.set_ylabel('diameter', size=15)         
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    sex = abalone_df.to_numpy()[:,0]
    rings = abalone_df.to_numpy()[:,-1]
    ax.set_title("Rings Binned Data", size=20)
    ax.bar(sex, rings, color='red')
    ax.grid('on')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['F', 'I', 'M'])
    ax.set_xlabel('sex', size=15)
    ax.set_ylabel('rings', size=15)         
    st.pyplot(fig)

with col3:
    st.write('**Dataset Structure**')
    st.dataframe(abalone_df)

# Dataframe description
st.write('Dataset Description')
st.table(abalone_df.describe())

# Correlation Matrix
_, col2, _ = st.columns(3)
with col2:
    fig, ax = plt.subplots(figsize=(5,5))
    columns = abalone_df.columns[:-1]
    values = abalone_df[columns].corr().values
    im = ax.imshow(values)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(columns, size=12)
    ax.set_yticklabels(columns, size=12)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, np.round(values[i, j],2), ha="center", va="center", color="w")
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    st.pyplot(fig)