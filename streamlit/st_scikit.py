import pandas as pd
import joblib
import keras
import numpy as np
import streamlit as st

from PIL import Image
from utils_scikit import *
from streamlit_drawable_canvas import st_canvas

merge_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = joblib.load("./models/random_forest.joblib")   
nn_model = keras.models.load_model('./models/2emnist_save.h5')

if 'draw_update' not in st.session_state:
    st.session_state['draw_update'] = False

      
def main():
    tab1, tab2, tab3 = st.tabs(['Canvas', 'Camera', 'Instructions'])
    
    # Canvas tab
    with tab1:
        if st.button('predict'):
            st.session_state['draw_update'] = True
            # st.balloons()

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=10,
            stroke_color='white',
            background_color='black',
            update_streamlit = st.session_state['draw_update'],
            height=150,
            width=150,
            drawing_mode='freedraw',#drawing_mode,
            key="canvas",
            )

        #Update results when button is pressed and something is drawn on the canvas
        if st.session_state['draw_update'] == True: 
            if np.sum(canvas_result.image_data) > 5737500:
                print(np.sum(canvas_result.image_data))
                im = canvas_result.image_data
                pred_img = prepping(im)
                pred_img = pred_img.reshape(-1, 784)

                st.markdown(f"### Randomforest Predicted Letter: { merge_map[model.predict(pred_img)[0]] }")
                st.markdown(f"### CNN Predicted Letter: {get_nn_result(nn_model, ~canvas_result.image_data, merge_map, get_pic)}")
                # st.image(pred_img.reshape(28,28)*255, width=150)

                st.session_state['draw_update'] = False

    # Camera tab
    with tab2:
        col1, col2 = st.columns([5,1.7])
        with col1:
            camera_pic = st.camera_input("Take a picture to predict")

            if camera_pic is not None:
                threshold = st.slider("White threshold", 0.3, 1.0, 0.5)            
                img = np.asarray(Image.open(BytesIO(camera_pic.getbuffer())))
                im = preprocess_camera_picture(img, threshold=threshold)
                
                pred_img = prepping(im)
                pred_img = pred_img.reshape(-1, 784)

                st.markdown(f"### Randomforest Predicted Letter: { merge_map[model.predict(pred_img)[0]] }")
                st.markdown(f"### CNN Predicted Letter: {get_nn_result(nn_model, ~im, merge_map, get_pic)}")

        print(type(get_pic))

        with col2:
            if camera_pic is not None:
                # st.image(pred_img.reshape(28,28)*255, width=150)
                st.image(im)

    # Instruction tab
    with tab3:
        st.markdown(f'### App')
        st.markdown(f'The app uses both classical machine learning and neural networks to interpeperate capital letters from pictures.' )
        st.markdown(f'#### ○ Inputs')
        st.markdown(f'It uses camera or the mouse as input. Depending on active tab.' )
        st.markdown(f'#### ○ Outputs')
        st.markdown(f'Outputs two written lines where one result is from a random forest model and the other is a result from a neural network.' )
        st.markdown(f'#### ○ Canvas')
        st.markdown(f'Uses the mouse to input capital letters. The arrows back and forth are undo and redo command. While the trashcan icon empties the canvas area.' )
        st.markdown(f'#### ○ Camera')
        st.markdown(f'Uses the camera to take a picture. Make sure that there is no other shapes within the picture.' )
        st.markdown(f'In its current shape the printed letters needs to be written with a bold font in a white background' )
        st.markdown(f'The picture also needs to be taken under well lit conditions. The threshold slider can compensate for some of the light conditions. Its output is shown beside the camera window.' )

# Call main routine
if __name__ == '__main__':
    st.title('Handwritten letter predictor')
    main()