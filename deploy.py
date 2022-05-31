import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics 

import matplotlib.pyplot as plt

from joblib import load

clf = load('titanic_model.joblib')

import streamlit as st

st.image('logo_dsu.png', width = 300)

st.title('Spring 2022 - Titanic Project')
st.subheader('See what your chances of survival are!')


Pclass_input = st.selectbox(
     'What class are your tickets for?',
     ('First Class', 'Second Class', 'Third Class'))

sex_input = st.selectbox(
     'What is your sex?',
     ('Female', 'Male'))

age_input = st.number_input('What is your age?',
                           value = 0,
                           step = 1)

sibs_input = st.number_input('How many siblings / spouses do you have?',
                           value = 0,
                           step = 1)

parch_input = st.number_input('How many parents / children do you have?',
                           value = 0,
                           step = 1)

fare_input = st.number_input('What was your fare?',
                           value = 0.0,
                           min_value = 0.0,
                           max_value = 500.0,
                           step = .1)

port_input = st.selectbox(
     'What port did you embark from?',
     ('Cherbourg', 'Queenstown', 'Southampton'))

if st.button('Compute Likliehood of Survival!'):
    pclass = 0
    if Pclass_input == 'First Class':
        pclass = 1
    elif Pclass_input == 'Second Class':
        pclass = 2
    elif Pclass_input == 'Third Class':
        pclass = 3
    
    sex = 0
    if sex_input == 'Female':
        sex = 1
    else:
        sex = 0
        
    age = age_input
    
    sibs = sibs_input
    
    parch = parch_input
    
    fare = fare_input
    
    port = 0
    if port_input == 'Cherbourg':
        port = 0
    elif port_input == 'Queenstown':
        port = 3
    elif port_input == 'Southampton':
        port = 1
    
    prediction = clf.predict_proba([[pclass, sex, age, sibs, parch, fare, port]])[0]
    plt.pie(prediction, colors = ['red', 'green'])
    
    st.write(f"We think that your probability of survival would be {prediction[1]}.")
    st.pyplot(plt.gcf())