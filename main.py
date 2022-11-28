import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model , model_from_json
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
import imutils
from PIL import Image

prediction_classes = {0:'chicken_noodle',
                     1: 'dumplings',
                     2: 'fried_chicken',
                     3: 'fried_chicken_salad_sticky_rice',
                     4: 'fried_pork_curry_rice',
                     5: 'grilled_pork_with_sticky_rice',
                     6: 'lek_tom_yam',
                     7: 'mama_namtok',
                     8: 'pork_blood_soup',
                     9: 'pork_congee',
                     10: 'pork_suki',
                     11: 'rice_scramble_egg',
                     12: 'rice_topped_with_stir_fried_pork_and_basil',
                     13: 'rice_with_roasted_pork',
                     14: 'roasted_red_pork_noodle',
                     15: 'sliced_grilled_pork_salad',
                     16: 'steamed_rice_with_chicken',
                     17: 'steamed_rice_with_fried_chicken',
                     18: 'stir_fried_rice_noodles_with_chicken',
                     19: 'stir_fried_rice_noodles_with_soy_sauce_and_pork'}

st.header('Upload/Take an image')
uploaded_file = st.file_uploader("Upload/Take an image", type=["jpg","jpeg"])
if uploaded_file is None:
    st.stop()
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.save("img.jpg")
    img_arr = image.load_img("img.jpg", target_size=(224, 224))


st.image(img_arr)
x = image.img_to_array(img_arr)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)



# load json and create model
select_model = st.radio("Select model",('ResNet152V2_model', 'VGG16_model', 'VGG19_model'))

json_file = open(f'{select_model}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"{select_model}.h5")
print("Loaded model from disk")


y_pred=loaded_model.predict(x,batch_size=1)
st.write(y_pred)
y_pred = np.argmax(y_pred)
st.write(prediction_classes[y_pred])