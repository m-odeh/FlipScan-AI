import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


#Function for f1 score for keras
from keras import backend as K
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# Load the pre-trained MobileNet model with custom metric function
model = load_model('C:/Users/M-ODE/Desktop/Apziva/projects/4th Project/models/model-mobilenet.h5', custom_objects={'get_f1': get_f1})

# Function to preprocess and predict
def predict_image(img):
    # Convert PIL image to NumPy array
    img = img.resize((150, 150))  # Resize image to the model's expected size
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Make a prediction
    # Make a prediction
    prediction = model.predict(img_array)
    probability_not_flip = prediction[0, 0] * 100  # Probability for "Not Flip"
    probability_flip = 100 - probability_not_flip  # Probability for "Flip"
    
    # Convert the prediction to a class label
    class_label = "Not Flip" if probability_not_flip > 50 else "Flip"
    if class_label == "Not Flip":
        final_probability=probability_not_flip
    elif class_label == "Flip":
        final_probability=probability_flip
        
    return class_label, final_probability



# Define sample images
sample_images = [
    "sample/sample1.jpg",
   "sample/sample2.jpg",
    "sample/sample3.jpg",
    "sample/sample4.jpg",
    "sample/sample5.jpg",
    "sample/sample6.jpg",
    "sample/sample7.jpg",
    "sample/sample8.jpg",
]

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Class Label"), gr.Textbox(label="Probability")],
    examples=[[sample_image] for sample_image in sample_images],
    live=True,
    title="Flipping Page Detector",
    description="Upload an image to classify whether it is a flip or not ",
)

# Launch the Gradio interface
iface.launch()