import streamlit as st
import pandas as pd
import joblib
import os

# إعداد الواجهة
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection")
st.markdown("### ✍️ Enter Text for Prediction")

# قائمة النماذج المتاحة
Model_of_Classifier = {
    'NaiveBayes.pkl': 0.94,
    'LogisticRegression.pkl': 0.98,
    'DecisionTree.pkl': 0.99,
    'SupportVectorMachine.pkl': 0.99
}

# Sidebar - اختيار النموذج
st.sidebar.header("🛠️ Settings")
selected_model = st.sidebar.selectbox("Choose a Classification Model:", options=list(Model_of_Classifier.keys()))
model_accuracy = Model_of_Classifier[selected_model]
st.sidebar.markdown(f"### Model Accuracy\n**{model_accuracy * 100:.2f}%**")

# تحميل النموذج والـ vectorizer
vectorizer_path = "vectorizer.pkl"

if os.path.exists(selected_model) and os.path.exists(vectorizer_path):
    model = joblib.load(selected_model)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("Model or vectorizer file not found. Please upload them.")
    st.stop()

# الإدخال الفردي
text_input = st.text_area("Input News Text:")

if st.button("🚀 Predict"):
    if text_input:
        vect_text = vectorizer.transform([text_input])
        prediction = model.predict(vect_text)[0]
        result = "✅ REAL News" if prediction == 1 else "❌ FAKE News"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")

# رفع ملف CSV للتنبؤ بالجملة
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

# زر إعادة التشغيل
if st.sidebar.button("🔄 Reset"):
    st.rerun()
