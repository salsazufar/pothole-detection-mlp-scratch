import streamlit as st
import numpy as np
from PIL import Image
import os
import joblib

# Define constants
IMAGE_SIZE = 32

# Define the SimpleMLP class
class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_data):
        self.hidden_layer_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        predictions = self.sigmoid(self.output_layer_input)
        return predictions

    def predict(self, input_image):
        prediction = self.forward(input_image)
        return "Pothole" if prediction >= 0.5 else "Normal"

# Function to load the model
def load_model(model, file_path):
    model_data = joblib.load(file_path)
    model.weights_input_hidden = model_data['weights_input_hidden']
    model.bias_hidden = model_data['bias_hidden']
    model.weights_hidden_output = model_data['weights_hidden_output']
    model.bias_output = model_data['bias_output']
    print(f"Model loaded from {file_path}")

# Load the pre-trained model
model_path = 'model/mlp_model_78.joblib'
if os.path.exists(model_path):
    model = SimpleMLP(input_size=IMAGE_SIZE*IMAGE_SIZE, hidden_size=100, output_size=1)
    load_model(model, model_path)
else:
    st.error("Model not found. Please ensure the model file is in the correct path.")
    st.stop()

# Function to preprocess the image
def preprocess_image(image):
    img = image.convert('L')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img).flatten() / 255.0
    return img_array

# Set page configuration
st.set_page_config(page_title="Pothole Detection App", page_icon="phalt texture or something related to roads.")

# Inject custom CSS
st.markdown(
    """
    <style>
    .custom-title {
        text-align: center;
        font-size: 48px;
        color: #31748f;
    }
    .custom-subtitle {
        text-align: center;
        font-size: 20px;
        color: #5e5e5e;
    }
    .custom-image {
        display: flex;
        justify-content: center;
    }
    .custom-button {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
def main():
    st.markdown("<h1 class='custom-title'>Pothole Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='custom-subtitle'>Upload an image to check if it contains a pothole.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="fileUploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("Classifying...")

        # Preprocess the image
        input_image = preprocess_image(image)
        input_image = input_image.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_image)
        st.write(f"Prediction: {prediction}")

        # Display result with styling
        if prediction == "Pothole":
            st.error("This image contains a pothole.")
        else:
            st.success("This image does not contain a pothole.")

    # Add a footer
    st.markdown("<footer style='text-align: center; color: #8e8e8e;'>Created by Salsa Zufar Radinka Akmal                               </footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()