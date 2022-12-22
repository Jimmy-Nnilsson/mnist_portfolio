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


merge_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = joblib.load("./models/random_forest.joblib")   
nn_model = keras.models.load_model('./models/2emnist_save.h5')
      
def main():
    tab1, tab2 = st.tabs(['Canvas', 'Camera'])
    with tab1:
        st.write(' test test')
        
        if st.button('predict'):
            st.session_state['draw_update'] = True
            st.balloons()
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
                st.markdown(f"## Randomforest Predicted Letter: { merge_map[model.predict(pred_img)[0]] }")

                st.markdown(f"### CNN Predicted Letter: {get_nn_result(nn_model, ~canvas_result.image_data, merge_map, get_pic)}")


                st.session_state['draw_update'] = False
    with tab2:
        camera_pic = st.camera_input("Take a picture")
        if camera_pic is not None:
                    print(np.sum(camera_pic))
                    im = np.asarray(Image.open(BytesIO(camera_pic.getbuffer())))
                    print(im.shape)
                    pred_img = prepping(im, np.median(im))
                    print(pred_img.shape, np.sum(pred_img))
                    pred_img = pred_img.reshape(-1, 784)

                    st.markdown(f"## Randomforest Predicted Letter: { merge_map[model.predict(pred_img)[0]] }")
                    st.markdown(f"### CNN Predicted Letter: {get_nn_result(nn_model, camera_pic, merge_map, get_pic)}")
                    
                    st.image(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), clamp=True)
                    # st.image(get_pic(im, 2,np.median(im)*1.33))
                    # st.image(pred_img.reshape(28,28)*255)
                    # st.write(np.unique(im, return_counts=True))


if __name__ == '__main__':
    st.title('Handwritten letter predictor')
    main()