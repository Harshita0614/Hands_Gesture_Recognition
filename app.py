import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model_path = r"C:\Users\Harshita Mehta\handgesture\handgestures.h5"  # Update with your actual model path
model = load_model(model_path)

# Parameters
image_size = (64, 64)
categories =['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']  # Replace with your actual categories


# Function to predict gesture
def predict_gesture(frame, model):
    img = cv2.resize(frame, image_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    return categories[np.argmax(predictions[0])]

# Function to capture webcam image
def webcam_feed():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Flip frame for mirror effect
        cap.release()
        return frame
    cap.release()
    return None

# Streamlit UI/UX

st.title("Real-Time Gesture Recognition App")
st.write("This app detects hand gestures in real-time using your webcam feed.")

# Start webcam feed button
if st.button('Start Webcam'):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame.")
                break
                
            # Flip frame for a mirror-like effect
            frame = cv2.flip(frame, 1)
            
            # Define the region of interest (ROI) for gesture detection
            roi = frame[100:300, 100:300]  # Cropping the gesture region
            
            # Predict gesture in the ROI
            gesture = predict_gesture(roi, model)
            
            # Draw the ROI on the frame
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(frame, f'Gesture: {gesture}', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display the frame with the predicted gesture
            stframe.image(frame, channels="BGR")
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()

st.write("Press the 'Start Webcam' button to begin gesture recognition.")
