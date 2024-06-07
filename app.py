import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import cv2
import json

# Load your emotion classification model
def load_model(model_path, weights_path):
    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

# Function to preprocess image for prediction
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict emotion of the image
def predict_emotion_single_image(img, model):
    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Function for real-time emotion detection using webcam
def real_time_emotion_detection(model, face_cascade):
    st.write("Real-time Facial Emotion Detection")
    cap = cv2.VideoCapture(0)
    stframe = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            predicted_class = predict_emotion_single_image(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)), model)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, class_names[predicted_class], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        stframe.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load your emotion classification model
model = load_model('model.json', 'model_weights.h5')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class_names = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Main Streamlit function
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
        real_time_emotion_detection(model, face_cascade)

    elif option == "Emotion Analysis for Local Images":
        st.write("Emotion Analysis for a Single Image")
        single_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if single_image is not None:
            img = Image.open(single_image)
            predicted_emotion = predict_emotion_single_image(img, model)
            st.write("Predicted Emotion for Uploaded Image:", class_names[predicted_emotion])

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
