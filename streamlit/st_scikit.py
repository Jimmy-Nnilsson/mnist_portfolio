import pandas as pd
from PIL import Image
import streamlit as st
from utils_scikit import *
import numpy as np
from streamlit_drawable_canvas import st_canvas
import random
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# Specify canvas parameters in application
import keras
import keras.backend as K

from pathlib import Path

merge_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = joblib.load("../models/random_forest.joblib")   
      
def main():
    if st.button('predict'):
        st.session_state['draw_update'] = True
    drawing_mode = 'freedraw'

    if 'draw_update' not in st.session_state:
        st.session_state['draw_update'] = False

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color='white',
        background_color='black',
        update_streamlit = st.session_state['draw_update'],
        height=150,
        width=150,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if st.session_state['draw_update'] == True: 
        print("inside")
        if np.sum(canvas_result.image_data) > 5737500:
            print(np.sum(canvas_result.image_data))
            im = canvas_result.image_data
            print(im.shape)
            pred_img = prepping(im)
            print(pred_img.shape, np.sum(pred_img))
            pred_img = pred_img.reshape(-1, 784)
            st.markdown(f"## Predicted number: { merge_map[model.predict(pred_img)[0]] }")
            st.session_state['draw_update'] = False

if __name__ == '__main__':
    st.title('Handwritten letter predictor')
    main()