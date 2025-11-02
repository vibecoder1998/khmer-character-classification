import streamlit as st
from PIL import Image
from src.preprocess import preprocess_image
from src.inference import load_model_and_encoder, predict
from utils.label_map import LABEL_TO_KHMER 

st.set_page_config(page_title="Khmer Character Classifier 🇰🇭", page_icon="🇰🇭")
st.title("🧠 Khmer Character Classifier")
st.write("Upload a **Khmer character** image to see what it predicts!")

# Add some information about supported characters
with st.expander("ℹ️ About this classifier", expanded=True):
    st.write("""
    This model can recognize the following Khmer consonants:
    - ក, គ, ខ, ង, ត, ដ, ណ, ឆ, ជ, ឈ
    """)

uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        
        # Show uploaded image first
        st.image(image, caption="Uploaded Image", use_container_width=False, width=300)
        
        with st.spinner("Classifying character... ⏳"):
            # Load model and make prediction
            model, le = load_model_and_encoder()
            x = preprocess_image(image)
            label_const, confidence, probs = predict(model, le, x)
            
            # 🔡 Map English label constant to Khmer character
            khmer_char = LABEL_TO_KHMER.get(label_const)
            
            # Get top predictions for additional insights
            top_k = 3
            top_indices = probs.argsort()[-top_k:][::-1]
            top_labels = le.inverse_transform(top_indices)
            top_confidences = probs[top_indices]
            
            # Handle low confidence predictions
            if confidence < 0.5:
                st.warning("⚠️ Low confidence prediction. The image might be unclear or not a Khmer character.")

        # Display prediction results
        st.success("✅ Prediction Complete!")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Prediction Result")
            st.markdown(f"**English Label:** `{label_const}`")
            st.markdown(f"**Khmer Character:** `{khmer_char}`")
            st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
            
        with col2:
            # Visual confidence meter
            st.markdown("### 📊 Confidence Level")
            st.progress(float(confidence))
            if confidence > 0.8:
                st.success("High confidence 🎯")
            elif confidence > 0.5:
                st.info("Moderate confidence 👍")
            else:
                st.warning("Low confidence 🤔")
        
        # Show the predicted Khmer character in large font
        st.markdown("---")
        st.markdown(f"# 🎯 Predicted Character: {khmer_char}")
        st.markdown(f"*({label_const})*")
        
        # Show top 3 predictions for comparison
        st.markdown("### 🔝 Top 3 Predictions")
        for i, (label, conf) in enumerate(zip(top_labels, top_confidences)):
            khmer_char_top = LABEL_TO_KHMER.get(label)
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.write(f"{emoji} **{khmer_char_top}** ({label}): `{conf * 100:.2f}%`")
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Tips: Make sure the image contains a clear Khmer character on a clean background.")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using PyTorch & Streamlit")