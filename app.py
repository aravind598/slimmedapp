from functools import cache
import streamlit as st
#from streamlit_tensorboard import st_tensorboard
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ExifTags
import io
import requests
from img_classifier import getOutput, prepare_my, make_my_prediction, make_prediction
import traceback
import copy
import time
import base64
import json
#import cv2
#from img_classifier import our_image_classifier
# import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

global model; global my_model; global tflite_colab; 
#global tflite_model_uint8; global tflite_model 
#global picture
global uri
global sentence
uri = None
sentence = None


@st.experimental_memo
@st.cache
@cache
def cache_image(image_byte: bytes, azure = False, img_shape: int = 224, upload = False, label = False) -> bytes:
    """[Cache the image and makes the image smaller before doing stuff]

    Args:
        image_byte (bytes): [Get from reading the file/image using image.read() or file.read()]
        img_shape (int, optional): [size of the default prediction/image tensor i.e needs to be 224x224 in image size for all my models]. Defaults to 224.

    Returns:
        bytes: [return a new bytes object that is smaller/faster to interpret]
    """
    #import piexif
    byteImgIO = io.BytesIO()
    image = Image.open(io.BytesIO(image_byte)).convert('RGB')
    try:
        #st.write(image.getexif())
        image = ImageOps.exif_transpose(image)
    except:
        pass
    #print(image.size)  
    size = (img_shape, img_shape)
    
    # Maintain Aspect Ratio of images by adding padding of black bars
    #image = ImageOps.pad(image, size, Image.ANTIALIAS)
    # Cut images to size by cropping into them
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    #print(image.size)
    
    #Lower the image size by decreasing its quality
    image.save(byteImgIO, format = "JPEG", optimize=True,quality = 90)
   
   #If the azure variable is True then dump the data as encoded utf-8 json for sending to the server 
    if azure:
        img_byte = byteImgIO.getvalue()  # bytes
        img_base64 = base64.b64encode(img_byte)  # Base64-encoded bytes * not str
        img_str = img_base64.decode('utf-8')  # str
        data = {"inference": img_str}
        return bytes(json.dumps(data),encoding='utf-8')
    
    if upload and label:
        img_byte = byteImgIO.getvalue()  # bytes
        img_base64 = base64.b64encode(img_byte)  # Base64-encoded bytes * not str
        img_str = img_base64.decode('utf-8')  # str
        data = {"label": label,
            "inference": img_str}
        return bytes(json.dumps(data),encoding='utf-8')
        
    
    byteImgIO.seek(0)
    image = byteImgIO.read()
    return image

@cache
def main():
    """
    [Main function all the UI is here]
    """
    # Metadata for the web app
    st.set_page_config(
    page_title = "WebApp",
    layout = "wide",
    page_icon= ":dog:",
    initial_sidebar_state = "collapsed",
    )


    
    menu = ['Home', 'Cloud Training']
    choice = st.sidebar.selectbox("Menu", menu)
        
    button_string = ""
    Flasking = False
    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title("Aravind's Application")
        # Now setting up a header text
        #st.subheader("By Your Cool Dev Name")
        
        #Expander 1
        my_expandering = st.expander(label='Model URL Input')
        with my_expandering:
            try:
                uri = st.text_input('Enter Azure ML/Flask Inference URL here:')
            except:
                uri = ""
        if uri:
            if 'grok' in uri:
                st.write("Flask Inference URL: " + uri)
                Flasking = True
                button_string = "Flask ML Predict"
            else:
                st.write("Azure ML Inference URL: " + uri)
                button_string = "Azure ML Predict"
        
        #Expander 1.5
        #QR code input but need to manually copy and paste into the above line to store the uri
        #my_expanders = st.expander(label="QR Code Input")
        
        checking_list = ["http"]
        
        #Expander 2
        #Changed from this
        #sentence = st.text_input('Input your sentence here:')
        # To this using an expander
        my_expander = st.expander(label='Inference for images on the internet:')
        image_bytes = None
        with my_expander:
            sentence = st.text_input('Input your image url here:') 
            if sentence:
                try:
                    response = requests.get(sentence)
                    # = Image.open(io.BytesIO(response.content))
                    image_bytes = response.content
                    #st.write(str(sentence))
                except Exception as e:
                    st.error("Exception occured due to url not having image " + str(e))
                    image = None
                    #st.error()
                    


        #Expander 3
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        
        #Copy images if not error
        uploaded_file1 = copy.copy(uploaded_file)
        uploaded_copy = copy.copy(uploaded_file)
        image1 = copy.copy(image_bytes)
        #picture1 = copy.copy(picture)

        jsonImage = None
        if uploaded_file is not None:
            upload = uploaded_file.read()
            jsonImage = cache_image(image_byte=upload, azure=True)
        elif image_bytes is not None:
            jsonImage = cache_image(image_byte=image1, azure=True)
            

            
        if uploaded_file is not None:
            placeholder = st.image(uploaded_file1.read(),use_column_width=True)
        elif image_bytes is not None:
            placeholder = st.image(image1,use_column_width=True)
        #elif picture is not None:
            #placeholder = st.image(picture1.read(),use_column_width=True)
        else:
            pass
        
        #azpredictbut = st.button("Azure ML Predict")
        try:
            if uri:
                if all(x in uri for x in checking_list):
                    #st.write(str(uriparts in uri for uriparts in checking_list))
                    if st.button(button_string):
                            if uploaded_file is not None or image_bytes is not None:
                                t = time.time()
                                if jsonImage:
                                    if Flasking:
                                    ##### Flask Colab ###################################################################################################################################################
                                        headers = {
                                            'Content-type': 'application/json', 'Accept': 'text/plain'}
                                        response = requests.post(
                                            uri + '/predict', data=jsonImage, headers=headers)
                                        label = str(response.text)
                                else:
                                    label = "error"
                                    st.error("JsonImage error")
                                    
                                st.success(label)
                                st.success("Time taken is : " + str(time.time()- t))
                            
                            else:
                                st.error("Can you please upload an image 🙇🏽‍♂️")
                    else:
                        pass#st.error("Button not pressed")
                else:
                    uri = ""
                    st.error("URL is not correct/valid or empty. Did you include the /score? ")
            else:
                pass
                #st.error("URL is empty.")
        except Exception as e:
            st.error('Service unavailable -> Is the Server Running?')
            st.error(str(e))

        if st.button("Clear Screen"):
            placeholder.empty()
            
    elif choice == "Cloud Training":
        # Let's set the title of our Contact Page
        st.title('Cloud Training')
        
        button_string = None
        uri = None
        label = None
        
        my_expandering = st.expander(label='Model URL Input')
        with my_expandering:
            try:
                uri = st.text_input('Enter Azure ML/Flask Inference URL here:')
            except:
                uri = ""
        if uri:
            if 'grok' in uri:
                st.write("Flask Inference URL: " + uri)
                Flasking = True
                button_string = "Flask ML Training"
            else:
                st.write("Azure ML Inference URL: " + uri)
                button_string = "Azure ML Training"
        
        image_train = st.file_uploader(label = "Please upload the image for retraining", type=['jpeg','png','jpg'])
        
        if image_train is not None:
            label = st.radio(
                "What is the Image's Label",
                ('Dog', 'Cat', 'Person'))
            
            st.write(label)
            
    
        if image_train is not None:
            if uri and label:
                if st.button(button_string):
                    jsonImage = cache_image(image_byte=image_train.read(), label = label, upload = True)
                    headers = {'Content-type': 'application/json',
                               'Accept': 'text/plain'}
                    response = requests.post(uri + '/train', data=jsonImage, headers=headers)
                    label = str(response.text)
                    st.success("Label + Image has been sent to the cloud")
            
if __name__ == "__main__":
    main()
