import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import google.generativeai as genai

# Set up Gemini API key
GOOGLE_API_KEY = "AIzaSyDeOU6vNwAo3HflrO6ETjXvApqI2VK9-6Q"
genai.configure(api_key=GOOGLE_API_KEY)

# Load the trained model
model = load_model('my_model.keras')

# Define class names for the diseases
class_names = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf',
    'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf',
    'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf',
    'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf',
    'Soyabean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
    'Tomato leaf bacterial spot', 'Tomato leaf late blight',
    'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
    'grape leaf', 'grape leaf black rot'
]

# Function to get remedies from the Gemini API
def get_remedies(predicted_class):
    try:
        # Generate remedies using Gemini API
        prompt = f"What are the remedies for the disease identified as {predicted_class} in plants?"
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while fetching remedies: {e}"

# Streamlit app
st.title("Plant Disease Classification")
st.write("Upload an image of a plant leaf to classify the disease and get remedies.")

# Upload image
uploaded_file = st.file_uploader("Choose a plant leaf image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Predict the label
    label = model.predict(img_array)

    # Determine the predicted class
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]

    # Display the predicted class
    st.write(f"Predicted Disease: {predicted_class}")

    # Fetch remedies from the Gemini API
    remedies = get_remedies(predicted_class)

    # Display remedies
    st.write(f"Remedies for {predicted_class}:")
    st.text(remedies)

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Plot the image with the predicted label
    fig, ax = plt.subplots()
    ax.imshow(image.load_img(uploaded_file))
    ax.set_title(predicted_class)
    ax.axis('off')
    st.pyplot(fig)
