import streamlit as st
import pandas as pd
import joblib
import os

# Title and description
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection")
st.markdown("### ✍️ Enter Text for Prediction")

# Load model
model_path = "NaiveBayes.pkl"
vectorizer_path = "vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("Model or vectorizer file not found. Please upload them.")
    st.stop()

# Input text for single prediction
text_input = st.text_area("Input News Text:")

if st.button("🚀 Predict"):
    if text_input:
        vect_text = vectorizer.transform([text_input])  # convert to 2D
        prediction = model.predict(vect_text)[0]
        result = "✅ REAL News" if prediction == 1 else "❌ FAKE News"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")

# File upload for batch prediction
st.sidebar.header("🛠️ Settings")
st.sidebar.markdown("**Choose a Classification Model:**")
st.sidebar.selectbox("Model", options=["NaiveBayes.pkl"], index=0)

st.sidebar.markdown("### Model Accuracy")
st.sidebar.markdown("**94.00%**")

st.sidebar.markdown("### Upload CSV file for bulk prediction (must have 'text' column):")
uploaded_file = st.sidebar.file_uploader("Browse files", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV file must contain a 'text' column.")
    else:
        X = vectorizer.transform(df['text'])
        predictions = model.predict(X)
        df['prediction'] = ['REAL' if p == 1 else 'FAKE' for p in predictions]
        st.write("### 🔍 Bulk Prediction Results")
        st.dataframe(df[['text', 'prediction']])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

# Reset button
if st.sidebar.button("🔄 Reset"):
    st.rerun()
