import pandas as pd
from PIL import Image
import streamlit as st
from utils import *
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Specify canvas parameters in application
import keras
import keras.backend as K

from pathlib import Path

merge_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = keras.models.load_model('../model/emnist_save.json')
# model = keras.models.load_model('./model/cnn_largetrain.h5')
    
    


def main():
    drawing_mode = 'freedraw'


    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color =  st.sidebar.color_picker("Stroke color hex: ")
    bg_color = "#eee"
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.sidebar.checkbox("Update in realtime", True)



    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=150,
        width=150,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)
    # if canvas_result.json_data is not None:
    #     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    #     for col in objects.select_dtypes(include=['object']).columns:
    #         objects[col] = objects[col].astype("str")
    #     st.dataframe(objects)
    if st.button('predict'):
        if canvas_result.image_data is not None:
                    st.markdown(f"## Predicted number: {merge_map[np.argmax(model.predict(convert_picture(canvas_result.image_data)))]}")
                    st.image(convert_picture(canvas_result.image_data).reshape(28,28))
                    # st.write(np.argmax(model.predict(convert_picture(canvas_result.image_data))))

if __name__ == '__main__':
    st.title('Handwritten number predictor')
    main()