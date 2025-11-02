import streamlit as st
from PIL import Image
from src.preprocess import preprocess_image
from src.inference import load_model_and_encoder, predict
from utils.label_map import LABEL_TO_KHMER  # ✅ Khmer label mapping

st.set_page_config(page_title="Khmer Character Classifier 🇰🇭", page_icon="🇰🇭")
st.title("🧠 Khmer Character Classification")
st.write("Upload a **Khmer character** image to see what it predicts!")

uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    with st.spinner("Predicting... ⏳"):
        model, le = load_model_and_encoder()
        x = preprocess_image(image)
        label_const, confidence, _ = predict(model, le, x)

        # 🔡 Map English label constant to Khmer character
        khmer_char = LABEL_TO_KHMER.get(label_const, label_const)

    # Display prediction **above the image**
    st.markdown(f"### 🔍 Predicted Character: **{label_const} → {khmer_char}**")
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence * 100:.2f}%")

    # Show uploaded image below prediction
    st.image(image, caption="Uploaded Image", use_container_width=False, width=600)
