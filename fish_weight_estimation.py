import time

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained machine learning model
with open('fish_weight_estimation_2.pkl', 'rb') as file:
    model = pickle.load(file)


with st.expander("about us "):
    st.write("The objective of this end-of-year project is to develop an artificial intelligence solution for modernizing fish farming.")
    st.write("The project aims to leverage computer vision techniques to facilitate the process of fish farming")

# Define the UI elements and callbacks
st.title('Fish Weight Predictor')
st.write('Enter the following information to predict the weight of a fish:')
fish_names={'Bream':0, 'Parkki':1, 'Perch':2, 'Pike':3, 'Roach':4, 'Smelt':5, 'Whitefish':6}

# Create the selectbox using the keys of the fish_names dictionary
species = st.selectbox('Species', list(fish_names.keys()))

# Get the numerical value for the selected fish species
species = fish_names[species]

length1 = st.number_input('Length (cm)')
height = st.number_input('Height (cm)')
# Define the prediction function
def predict_weight(species, length1, height):
    X = pd.DataFrame({
        'Species': [species],
        'Length1': [length1],
        'Height': [height]
    })
    return model.predict(X)[0]

# Define the 'Predict' button and its callback
if st.button('Predict'):
    weight = predict_weight(species, length1, height)
    if(species == 0 ):
        w = 0.01530 * length1 ** 0.958
    elif (species == 1 ):
        w = 0.01530 * length1 ** 1.958
    elif (species == 2 ):
        w = 0.01530 * length1 ** 2.958
    elif (species == 3 ):
        w = 0.01530 * length1 ** 3.958
    elif (species == 4 ):
        w = 0.01530 * length1 ** 4.958
    elif (species == 5 ):
        w = 0.01530 * length1 ** 5.958
    else :
        w = 0.01530 * length1 ** 6.958

    average = (weight + w)/2
    with st.spinner(text='Calculating the weight'):
        time.sleep(3)
        st.warning(f'The predicted weight of the fish is {weight:.2f} grams.')
        st.warning(f'The predicted weight of the fish with LWR is {w:.2f} grams.')
        st.success(f'The final predicted weight of the fish is {average:.2f} grams.')



