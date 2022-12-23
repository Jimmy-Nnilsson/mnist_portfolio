import pandas as pd
from PIL import Image
import streamlit as st
from utils import *
import joblib
from io import StringIO, BytesIO
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
# Specify canvas parameters in application
import keras
import keras.backend as K

from sklearn.pipeline import Pipeline
from io import BytesIO

from pathlib import Path

if 'canvas_update' not in st.session_state:
    st.session_state.canvas_update = False

my_pipeline = Pipeline([('transformed_dataset', EMNISTDataPreparation())])

merge_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = keras.models.load_model('../data_prep/model/2emnist_save.h5')
model2 = joblib.load('../data_prep/model/dd_sklearn.sav')
def main():
    col1, col2, col3 = st.columns((5,5,2))
    # Create a canvas component
    with col2:
        
        bg_image = st.file_uploader("uploader", type=["png", "jpg"], label_visibility='hidden')
        if bg_image: 
            st.markdown(f"### Predicted number: {get_nn_result(model, bg_image, merge_map, get_pic)}")
            st.image(my_pipeline.transform(bg_image)*255)
            

            
            img  = np.array(Image.open(BytesIO(bg_image.getbuffer())))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = ~img
            # img_offset = 255 - np.mean(img)
            # img + (img_offset * 0.2)
            # st.write(img_offset, np.mean(img))
            st.image(img, clamp=True)
            print(img)
            

            
        
    with col3:
        if bg_image: 
            st.image(bg_image)
    
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
            stroke_width=8,
            stroke_color="#000",
            background_color="#eee",
            update_streamlit=True,
            # update_streamlit=True,
            height=150,
            width=150,
            # background_image=Image.open(bg_image) if bg_image else None,
            drawing_mode='freedraw',
            key="canvas",
            )

        if canvas_result.image_data is not None and np.sum(canvas_result.image_data) < 21802500:
                st.markdown(f"### Predicted number: {get_nn_result(model, canvas_result.image_data, merge_map, get_pic)}")
                st.image(my_pipeline.transform(canvas_result.image_data)*255)
                # st.write(merge_map[np.argmax(model.predict(np.reshape(~my_pipeline.transform(canvas_result.image_data)*255, (1,28,28))))])
                # st.write(my_pipeline.transform(canvas_result.image_data).reshape(28*28))
                img = my_pipeline.transform(canvas_result.image_data).reshape(-1,28*28)
                st.write(merge_map[model2.predict(img)[0]])
                
                
    # with col3:
    #     if canvas_result.image_data is not None and np.sum(canvas_result.image_data) < 21802500 and bg_image:
    #         pic1 = get_pic(np.asarray(Image.open(BytesIO(bg_image.getbuffer()))), 2)
    #         pic2 = get_pic(canvas_result.image_data, 2)
    #         st.image(pic1, clamp=True)
    #         st.image(pic2, clamp=True)
    #         st.image(pic1 + pic2, clamp=True)
    # if st.button('predict'):
    #     if st.session_state.canvas_update:
    #         st.session_state.canvas_update = False
    #     else:
    #         st.session_state.canvas_update = True
    # st.write(st.session_state.canvas_update)



                    

if __name__ == '__main__':
    st.title('Handwritten number predictor')
    main()