


import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Available Classification Models and Accuracy
Model_of_Classifier = {
    'NaiveBayes.pkl': 0.94,
    'LogisticRegression.pkl': 0.98,
    'DecisionTree.pkl': 0.99,
    'SupportVectorMachine.pkl': 0.99
}

# Label Mapping
classification_labels = {
    0: "Fake News",
    1: "Real News",
}

# Features
Classification_features = ['text']

# Streamlit App
st.title("📰 Fake News Detection")
st.sidebar.header("🔧 Settings")

# Sidebar: Select Classification Model
selected_model = st.sidebar.selectbox("Choose a Classification Model:", list(Model_of_Classifier.keys()))
model_accuracy = Model_of_Classifier[selected_model]
st.sidebar.metric(label="Model Accuracy", value=f"{model_accuracy * 100:.2f}%")

# Load selected model
try:
    with open(selected_model, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file '{selected_model}' not found.")
    st.stop()

# Sidebar: Bulk CSV Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file for bulk prediction (must have 'text' column):", type=['csv'])

# Sidebar: Reset
if st.sidebar.button("🔄 Reset"):
    st.experimental_rerun()

# Compare Models
if st.sidebar.button("📊 Compare Models"):
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame({
        'Model': list(Model_of_Classifier.keys()),
        'Accuracy (%)': [acc * 100 for acc in Model_of_Classifier.values()]
    })
    st.dataframe(comparison_df)
    st.bar_chart(comparison_df.set_index('Model'))

# Input Section
if uploaded_file:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_data.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.subheader("✍️ Enter Text for Prediction")
    user_input = st.text_area("Input News Text:")

# Predict Button
if st.button("🚀 Predict"):
    try:
        if uploaded_file:
            if 'text' not in input_data.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                predictions = model.predict(input_data['text'])
                mapped_preds = [classification_labels.get(int(p), "Unknown") for p in predictions]
                input_data['Prediction'] = mapped_preds
                st.write("Predictions:")
                st.dataframe(input_data)

                # Visualize
                unique, counts = np.unique(mapped_preds, return_counts=True)
                hist_df = pd.DataFrame({'Class': unique, 'Count': counts})
                st.bar_chart(hist_df.set_index('Class'))

        else:
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                prediction = model.predict([user_input])
                predicted_label = classification_labels.get(int(prediction[0]), "Unknown")
                st.success(f"The Predicted Class is: **{predicted_label}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
