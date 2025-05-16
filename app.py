import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
from PIL import Image

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Parameters
img_size = 128
model_save_path = 'mobilenetv2_gesture_model.keras'
label_encoder_save_path = 'label_encoder.pkl'

# Load model and labels once
@st.cache_resource
def load_model_and_labels():
    model = load_model(model_save_path)
    with open(label_encoder_save_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_labels()

st.title("Gesture Recognition with MobileNetV2")
st.sidebar.title("Controls")

# Sidebar controls
mode = st.sidebar.radio("Choose Mode", ["Webcam", "Upload Image"])

# Webcam prediction
if mode == "Webcam":
    start_button = st.sidebar.button("Start Webcam")
    stop_button = st.sidebar.button("Stop Webcam")

    FRAME_WINDOW = st.image([])

    if start_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam")
        else:
            st.session_state['webcam_running'] = True
            while st.session_state.get('webcam_running', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image")
                    break

                # Preprocess the frame
                resized = cv2.resize(frame, (img_size, img_size))
                preprocessed = preprocess_input(resized)
                input_data = np.expand_dims(preprocessed, axis=0)

                # Predict the gesture
                preds = model.predict(input_data)
                pred_class = np.argmax(preds[0])
                pred_label = label_encoder[pred_class]

                # Overlay prediction on the frame
                cv2.putText(frame, f"Gesture: {pred_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Show the frame in Streamlit
                FRAME_WINDOW.image(frame)

                # Stop webcam if the "Stop Webcam" button is pressed
                if stop_button:
                    st.session_state['webcam_running'] = False

            cap.release()
            FRAME_WINDOW.image([])

# Image upload prediction
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        resized_image = image.resize((img_size, img_size))
        preprocessed_image = preprocess_input(np.array(resized_image))
        input_data = np.expand_dims(preprocessed_image, axis=0)

        # Predict the gesture
        preds = model.predict(input_data)
        pred_class = np.argmax(preds[0])
        pred_label = label_encoder[pred_class]

        st.write(f"**Predicted Gesture:** {pred_label}")


