import streamlit as st
import pandas as pd
from io import StringIO
from streamlit_image_select import image_select
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import gridspec
import functools
import os
#import cv2 
import tensorflow_hub as hub
import PIL
import time

#st.set_page_config(layout="wide")
with st.container(border=True):

    st.header('StyloGEN', divider = 'rainbow')
    st.subheader('_Style_ :blue[yourself] on the go :sunglasses:')


with st.container(border=True):
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Paste Link", "Live Capture"])

with tab1:

    with st.container(border=True):
        st.caption('Choose the :red[**_BASE_**] image')
        base_file = st.file_uploader(" ", type = ['jpeg'])
        
    with st.container(border=True):
        st.caption('Choose the :blue[**_STYLE_**] you want to apply')
        mask_image = image_select('',["./style_transfer/style_images/white.jpeg",
                                    "./style_transfer/style_images/monalisa.jpeg",
                                    "./style_transfer/style_images/starry_night.jpeg",
                                    "./style_transfer/style_images/scream.jpeg",
                                    "./style_transfer/style_images/girldance.jpeg",
                                    "./style_transfer/style_images/greatwave.jpeg",
                                    "./style_transfer/style_images/impression.jpeg",
                                    "./style_transfer/style_images/persistence.jpeg",
                                    "./style_transfer/style_images/efortvaux.jpeg",
                                    "./style_transfer/style_images/wanderer.jpeg",
                                    "./style_transfer/style_images/eninthwave.jpeg",
                                    "./style_transfer/style_images/gypsy.jpeg"],
                                    captions=["None","Monalisa", "Starry Night", "Scream",'Girldance',
                                                'Greatwave','Impression', 'Persistence','Fortvaux','Wanderer','NinthWave','Gypsy'])

        #st.caption('Choose the :blue[**_STYLE_**] you want to apply')
        #st.image(img)
    with st.container(border=True):

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.caption(':red[**_BASE IMAGE_**]')
                if base_file is not None:
                    original_image = base_file
                    st.image(base_file)
                    with open(os.path.join("tempDir",'download.jpeg'),"wb") as f: 
                        f.write(base_file.getbuffer())
                    st.success("Saved File")


        with col2:
                with st.container(border=True):
                    st.caption(':blue[**_STYLE IMAGE_**]')
                    if mask_image != "./style_transfer/style_images/white.jpeg":
                        st.image(mask_image)


    st.button("Reset", type="primary")

    if st.button('GENERATE', key = 'generate_upload'):
        with st.container(border=True):
            
            def image_cropper(image):
                image_shape = image.shape
                cropped_shape = min(image_shape[1], image_shape[2])
                offset_y = max(image_shape[1] - image_shape[2], 0) // 2
                offset_x = max(image_shape[2] - image_shape[1], 0) // 2
                cropped_image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, cropped_shape, cropped_shape)
                return cropped_image

            @functools.lru_cache(maxsize = None)

            def load_image(image_url, image_size = (256, 256), preserve_aspect_ratio = True):
                image_path = image_url
                image = tf.io.decode_image(tf.io.read_file(image_path), channels = 3, dtype = tf.float32)[tf.newaxis, ...]
                image = image_cropper(image)
                image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
                return image

            generated_image_size = 384
            original_image_size = (generated_image_size, generated_image_size)
            mask_image_size = (256, 256)
            mask_image_path = mask_image
            original_image_path = './tempDir/download.jpeg'

            original_image = load_image(original_image_path, original_image_size)
            mask_image = load_image(mask_image_path, mask_image_size)
            mask_image = tf.nn.avg_pool(mask_image, ksize = [3,3], strides = [1,1], padding = 'SAME')

            styler_module = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
            image_styler = hub.load(styler_module)

            outputs = image_styler(original_image, mask_image)
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            generated_image = outputs[0]

            def tensor_to_image(tensor):
                tensor = tensor*255
                tensor = np.array(tensor, dtype=np.uint8)
                if np.ndim(tensor)>3:
                    assert tensor.shape[0] == 1
                    tensor = tensor[0]
                return PIL.Image.fromarray(tensor)
            st.caption(':blue[**GENERATED IMAGE**]')
            st.success("Styled Image Generated !!!")
            st.image(tensor_to_image(generated_image))

            from io import BytesIO
            buf = BytesIO()
            tensor_to_image(generated_image).save(buf, format="JPEG")
            byte_im = buf.getvalue()            

            st.download_button(label='Download Image',data= byte_im,
                        file_name='generated_image.png',
                        mime='image/jpeg')
            
            
    

with tab2:

    with st.container(border=True):
        st.caption('Paste the :blue[**_BASE_**] image link')
        original_image_url = st.text_input('', 'base link')
        
    with st.container(border=True):
        st.caption('Paste the :red[**_STYLE_**] image link')
        mask_image_url = st.text_input('', 'style link')
        
        
    
    with st.container(border=True):

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.caption(':red[**_BASE IMAGE_**]')
                if original_image_url != 'base link':
                    st.image(original_image_url)
                


        with col2:
                with st.container(border=True):
                    st.caption(':blue[**_STYLE IMAGE_**]')
                    if mask_image_url != 'style link':
                        st.image(mask_image_url)

    st.button("Reset", type="primary", key = 'reset2')
    if st.button('GENERATE', key = 'generate_link'):
        with st.container(border=True):
            
            def image_cropper(image):
                image_shape = image.shape
                cropped_shape = min(image_shape[1], image_shape[2])
                offset_y = max(image_shape[1] - image_shape[2], 0) // 2
                offset_x = max(image_shape[2] - image_shape[1], 0) // 2
                cropped_image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, cropped_shape, cropped_shape)
                return cropped_image

            @functools.lru_cache(maxsize = None)

            def load_image(image_url, image_size = (256, 256), preserve_aspect_ratio = True):
                image_path = image_url
                image = tf.io.decode_image(tf.io.read_file(image_path), channels = 3, dtype = tf.float32)[tf.newaxis, ...]
                image = image_cropper(image)
                image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
                return image

            generated_image_size = 384
            original_image_size = (generated_image_size, generated_image_size)
            mask_image_size = (256, 256)
            original_image_url =  tf.keras.utils.get_file('original_image.jpeg', original_image_url)
            mask_image_url =  tf.keras.utils.get_file('mask_image.jpeg', mask_image_url)
            original_image = load_image(original_image_url, original_image_size)
            mask_image = load_image(mask_image_url, mask_image_size)
            mask_image = tf.nn.avg_pool(mask_image, ksize = [3,3], strides = [1,1], padding = 'SAME')

            styler_module = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
            image_styler = hub.load(styler_module)

            outputs = image_styler(original_image, mask_image)
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            generated_image = outputs[0]

            def tensor_to_image(tensor):
                tensor = tensor*255
                tensor = np.array(tensor, dtype=np.uint8)
                if np.ndim(tensor)>3:
                    assert tensor.shape[0] == 1
                    tensor = tensor[0]
                return PIL.Image.fromarray(tensor)
            st.caption(':blue[**GENERATED IMAGE**]')
            st.success("Styled Image Generated !!!")
            st.image(tensor_to_image(generated_image))

            from io import BytesIO
            buf = BytesIO()
            tensor_to_image(generated_image).save(buf, format="JPEG")
            byte_im = buf.getvalue()            

            st.download_button(label='Download Image',data= byte_im,
                        file_name='generated_image.png',
                        mime='image/jpeg')




with tab3:
    with st.container(border=True):

        st.caption('Capture a :blue[**_PHOTO_**]')
        base_file = st.camera_input(" ")

    with st.container(border=True):
        st.caption('Choose the :blue[**_STYLE_**] you want to apply')
        mask_image = image_select('',["./style_transfer/style_images/white.jpeg",
                                    "./style_transfer/style_images/monalisa.jpeg",
                                    "./style_transfer/style_images/starry_night.jpeg",
                                    "./style_transfer/style_images/scream.jpeg",
                                    "./style_transfer/style_images/girldance.jpeg",
                                    "./style_transfer/style_images/greatwave.jpeg",
                                    "./style_transfer/style_images/impression.jpeg",
                                    "./style_transfer/style_images/persistence.jpeg",
                                    "./style_transfer/style_images/efortvaux.jpeg",
                                    "./style_transfer/style_images/wanderer.jpeg",
                                    "./style_transfer/style_images/eninthwave.jpeg",
                                    "./style_transfer/style_images/gypsy.jpeg"],
                                    captions=["None","Monalisa", "Starry Night", "Scream",'Girldance',
                                                'Greatwave','Impression', 'Persistence','Fortvaux','Wanderer','NinthWave','Gypsy'], key='image_select2')

    with st.container(border=True):

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.caption(':red[**_BASE IMAGE_**]')
                if base_file is not None:
                    original_image = base_file
                    st.image(base_file)
                    with open(os.path.join("tempDir",'capture.jpeg'),"wb") as f: 
                        f.write(base_file.getbuffer())
                    st.success("Saved File")


        with col2:
                with st.container(border=True):
                    st.caption(':blue[**_STYLE IMAGE_**]')
                    if mask_image != "./style_transfer/style_images/white.jpeg":
                        st.image(mask_image)
    
    st.button("Reset", type="primary", key = 'reset3')

    if st.button('GENERATE', key = 'generate_capture'):
        with st.container(border=True):
            
            def image_cropper(image):
                image_shape = image.shape
                cropped_shape = min(image_shape[1], image_shape[2])
                offset_y = max(image_shape[1] - image_shape[2], 0) // 2
                offset_x = max(image_shape[2] - image_shape[1], 0) // 2
                cropped_image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, cropped_shape, cropped_shape)
                return cropped_image

            @functools.lru_cache(maxsize = None)

            def load_image(image_url, image_size = (256, 256), preserve_aspect_ratio = True):
                image_path = image_url
                image = tf.io.decode_image(tf.io.read_file(image_path), channels = 3, dtype = tf.float32)[tf.newaxis, ...]
                image = image_cropper(image)
                image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
                return image

            generated_image_size = 384
            original_image_size = (generated_image_size, generated_image_size)
            mask_image_size = (256, 256)
            mask_image_path = mask_image
            original_image_path = './tempDir/capture.jpeg'

            original_image = load_image(original_image_path, original_image_size)
            mask_image = load_image(mask_image_path, mask_image_size)
            mask_image = tf.nn.avg_pool(mask_image, ksize = [3,3], strides = [1,1], padding = 'SAME')

            styler_module = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
            image_styler = hub.load(styler_module)

            outputs = image_styler(original_image, mask_image)
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            generated_image = outputs[0]

            def tensor_to_image(tensor):
                tensor = tensor*255
                tensor = np.array(tensor, dtype=np.uint8)
                if np.ndim(tensor)>3:
                    assert tensor.shape[0] == 1
                    tensor = tensor[0]
                return PIL.Image.fromarray(tensor)
            st.caption(':blue[**GENERATED IMAGE**]')
            st.success("Styled Image Generated !!!")
            st.image(tensor_to_image(generated_image))

            from io import BytesIO
            buf = BytesIO()
            tensor_to_image(generated_image).save(buf, format="JPEG")
            byte_im = buf.getvalue()            

            st.download_button(label='Download Image',data= byte_im,
                        file_name='generated_image.png',
                        mime='image/jpeg')
            
