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
    3: "70",
    4: "80",
    5: "90",
    6: "End Of Limit",
    7: "100",
    8: "120"
}

others_mapping = {
    9: "Interdiction de dépasser",
    10: "interdiction de dépasser pour poids lourd",
    11: "Voie prioritaire",
    12: "priorité à droite",
    13: "Laisser passer",
    14: "Stop",
    15: "Interdit au véhicules",
    16: "Interdit au poids lourd",
    17: "Sens interdit",
    18: "Attention",
    19: "Attention virage à gauche",
    20: "Attention virage à droite",
    21: "Succession de virage",
    22: "Dos d'ane",
    23: "Risque dérapage",
    24: "Voie rétrécie",
    25: "Attention travaux",
    26: "Feu tricolore",
    27: "Attention piéton",
    28: "Attention enfants",
    29: "Attention vélo",
    30: "Attention gel",
    31: "Attention animaux",
    32: "Fin d'interdiction",
    33: "Tourner à droite",
    34: "Tourner à gauche",
    35: "Continuer tout droit",
    36: "Tout Droit ou Droite",
    37: "Tout Droit ou Gauche",
    38: "Placer vous à droite",
    39: "Placer vous à gauche",
    40: "Rond point",
    41: "Fin Interdiction de dépasser",
    42: "Fin interdiction de dépasser poids lourds"
}

# Cache the model loading so it runs only once per session
@st.cache_resource()
def load_models():
    binary_model = tf.keras.models.load_model('./modelsROI/best_binary_model.keras')
    whichSpeed_model = tf.keras.models.load_model('./modelsROI/best_whichSpeed_model.keras')
    whichSign_model = tf.keras.models.load_model('./modelsROI/best_whichSign_model.keras')
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
        actual_class = others_pred + 9
        predicted_label = others_mapping.get(actual_class, "Unknown Non-Speed Sign")
        result = f"Non-Speed Sign: {predicted_label}"
    
    return result

# --- Streamlit App ---

st.title("Traffic Sign Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Load the models
    binary_model, whichSpeed_model, whichSign_model = load_models()
    
    # Predict and display the result
    prediction = predict_sign(img, binary_model, whichSpeed_model, whichSign_model)
    
    st.subheader("Prediction")
    st.write(prediction)
