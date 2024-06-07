import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your emotion classification model
def load_model(model_path, weights_path):
    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

# Preprocessing function for the image
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load the model
model = load_model('model.json', 'model_weights.h5')

# Load the pre-trained face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class_names = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        i = self.i + 1

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img)
            preprocessed_img = preprocess_image(pil_img)
            predictions = model.predict(preprocessed_img)
            predicted_class = np.argmax(predictions, axis=1)[0]

            cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            cv2.putText(img, class_names[predicted_class], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img

def main():
    st.title("Facial Emotion Detection")
    st.sidebar.title("Dashboard")
    st.sidebar.markdown(
        """ Developed by Uday Singh    
            Model: EfficientnetV2  
            LinkedIn: https://www.linkedin.com/in/uday-singh-45563a209/""")
    option = st.sidebar.selectbox("Choose an option",
                                  ["Home", "Real-time Facial Emotion Analysis", "Emotion Analysis for Local Images",
                                   "About"])
    if option == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, EfficientNetV2 and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time facial emotion recognition.
                 
                 3. Emotion Prediction for local Images

                 """)

    elif option == "Real-time Facial Emotion Analysis":
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif option == "Emotion Analysis for Local Images":
        st.write("Emotion Analysis for a Single Image")
        single_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if single_image is not None:
            img = Image.open(single_image)
            preprocessed_img = preprocess_image(img)
            predictions = model.predict(preprocessed_img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            st.write("Predicted Emotion for Uploaded Image:", class_names[predicted_class])

    elif option == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="padding:10px">  Real time face emotion detection application using OpenCV, 
        Custom Trained CNN model and Streamlit. <br><br>
        This Application is developed by Uday Singh using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. 
        If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request.<br><br>
        Thanks for Visiting... </div> </br> """
        st.markdown(html_temp_about1, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
