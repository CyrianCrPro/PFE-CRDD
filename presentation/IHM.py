import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

# Define the mappings
speed_mapping = {
    0: "20",
    1: "30",
    2: "50",
    3: "60",
    4: "70",
    5: "80",
    6: "End Of Limit",
    7: "100",
    8: "120",
}

others_mapping = {
  1: "Interdiction de dépasser pour poids lourd",
  2: "Voie prioritaire",
  3: "priorité à droite",
  4: "Laisser passer",
  5: "Stop",
  6: "Interdit au véhicules",
  7: "Interdit au poids lourd",
  8: "Sens interdit",
  9: "Attention",
  10: "Attention virage à gauche",
  11: "Attention virage à droite",
  12: "Succession de virage",
  13: "Dos d'ane",
  14: "Risque dérapage",
  15: "Voie rétrécie",
  16: "Attention travaux",
  17: "Feu tricolore",
  18: "Attention piéton",
  19: "Attention enfants",
  20: "Attention vélo",
  21: "Attention gel",
  22: "Attention animaux",
  23: "Fin d'interdiction",
  24: "Tourner à droite",
  25: "Tourner à gauche",
  26: "Continuer tout droit",
  27: "Tout Droit ou Droite",
  28: "Tout Droit ou Gauche",
  29: "Placer vous à droite",
  30: "Placer vous à gauche",
  31: "Rond point",
  32: "Fin Interdiction de dépasser",
  33: "Fin interdiction de dépasser poids lourds",
  34: "Interdiction de dépasser"
}

# Cache the model loading so it runs only once per session
@st.cache_resource()
def load_models():
    binary_model = tf.keras.models.load_model('../models/best_binary_model.keras')
    whichSpeed_model = tf.keras.models.load_model('../models/best_whichSpeed_model.keras')
    whichSign_model = tf.keras.models.load_model('../models/best_whichSign_model.keras')
    return binary_model, whichSpeed_model, whichSign_model

def preprocess_image_pil(img, target_size=(100, 100)):
    """
    Resize and normalize the image as during training.
    """
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def predict_sign(img, binary_model, whichSpeed_model, whichSign_model):
    """
    Process the image through the binary classifier.
    Then, based on the binary result, use the appropriate model:
    - If the binary model predicts a speed sign (> 0.5), use whichSpeed_model.
    - Otherwise, use whichSign_model (with index adjustment) for non-speed signs.
    """
    img_batch = preprocess_image_pil(img, target_size=(100, 100))
    
    binary_pred_prob = binary_model.predict(img_batch)[0][0]
    
    if binary_pred_prob > 0.5:
        # Stage 2A: Predict which speed sign it is.
        speed_pred_probs = whichSpeed_model.predict(img_batch)[0]
        speed_pred = np.argmax(speed_pred_probs)
        predicted_label = speed_mapping.get(speed_pred, "Unknown Speed Sign")
        result = f"Speed Sign: {predicted_label}"
    else:
        # Stage 2B: Predict which non-speed sign it is.
        others_pred_probs = whichSign_model.predict(img_batch)[0]
        others_pred = np.argmax(others_pred_probs)
        # Adjust index: non-speed sign
        actual_class = others_pred + 1
        predicted_label = others_mapping.get(actual_class, "Unknown Non-Speed Sign")
        result = f"Non-Speed Sign: {predicted_label}"
    
    return result

# --- Streamlit App ---

st.title("Traffic Sign Recognition")

# Load the models
binary_model, whichSpeed_model, whichSign_model = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict and display the result
    prediction = predict_sign(img, binary_model, whichSpeed_model, whichSign_model)
    
    st.subheader("Prediction")
    st.write(prediction)
