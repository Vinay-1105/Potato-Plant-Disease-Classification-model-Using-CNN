import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# Permanent note: Tool scope
TOOL_SCOPE_NOTE = "Notice: This model is trained specifically on potato leaf images and is intended only for potato leaf photos."

# Class names exactly matching model training order
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
DISPLAY_NAMES = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight': 'Late Blight',
    'Potato___healthy': 'Healthy'
}

# Confidence threshold determined from validation set
CONFIDENCE_THRESHOLD = 0.70  # 70%

HEALTHY_WARNING = "A healthy result does not rule out disease. If you see spots or wilting, consult an agricultural expert."


@st.cache_resource
def load_classification_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_v2.keras")
    if not os.path.exists(model_path):
        model_path = "model_v2.keras"
    return tf.keras.models.load_model(model_path)


def predict(image, model):
    # Ensure RGB conversion to support PNG/RGBA and grayscale images
    rgb_img = image.convert('RGB')
    resized_img = rgb_img.resize((256, 256))
    
    # Raw float32 array in [0, 255]. Rescaling(1./255) is embedded inside the model.
    img_array = np.array(resized_img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_idx])
    predicted_class = CLASS_NAMES[pred_idx]

    return predicted_class, confidence, predictions


def main():
    st.set_page_config(
        page_title="Potato Leaf Disease Classifier",
        page_icon="🥔",
        layout="centered"
    )

    st.title("🥔 Potato Plant Disease Classifier")
    st.info(TOOL_SCOPE_NOTE)

    st.write(
        "Upload a close-up photograph of a potato leaf to check for **Early Blight**, **Late Blight**, or **Healthy** tissue."
    )

    uploaded_file = st.file_uploader(
        "Choose a potato leaf image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analyzing image..."):
                model = load_classification_model()
                pred_class, confidence, all_probs = predict(image, model)

            friendly_name = DISPLAY_NAMES.get(pred_class, pred_class)
            conf_percent = confidence * 100.0

            st.divider()

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ **Low Confidence Prediction:** The model predicted **{friendly_name}** with only "
                    f"**{conf_percent:.1f}%** confidence (below the {CONFIDENCE_THRESHOLD*100:.0f}% threshold). "
                    "This image may be ambiguous, out-of-distribution, or not a potato leaf."
                )
            else:
                if pred_class == 'Potato___healthy':
                    st.success(f"### Result: **{friendly_name}** ({conf_percent:.1f}% confidence)")
                    st.info(f"💡 {HEALTHY_WARNING}")
                else:
                    st.error(f"### Result: **{friendly_name}** ({conf_percent:.1f}% confidence)")

            # Detailed breakdown
            with st.expander("View Probability Breakdown"):
                for idx, c_name in enumerate(CLASS_NAMES):
                    prob = float(all_probs[idx]) * 100.0
                    label = DISPLAY_NAMES.get(c_name, c_name)
                    st.write(f"**{label}**: {prob:.2f}%")
                    st.progress(float(all_probs[idx]))

        except Exception as e:
            st.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main()