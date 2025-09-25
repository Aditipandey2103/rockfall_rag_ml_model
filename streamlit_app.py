import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Import your predictor class (RAG-enabled)
from rockfall_app import EnhancedRockfallPredictor
from rockfall_app import load_and_train
st.set_page_config(page_title="Rockfall Prediction System", layout="wide")
st.title("RAG-Enhanced Rockfall Prediction System")

# Sidebar for file upload and training
st.sidebar.header("Training Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV (with Risk column)", type=["csv"])
train_btn = st.sidebar.button("Train Models")

predictor = st.session_state.get("predictor") if "predictor" in st.session_state else None
if predictor is None:
    predictor = EnhancedRockfallPredictor()
    st.session_state["predictor"] = predictor

# --- Train models ---

if train_btn:
    if uploaded_file is None:
        st.sidebar.error('Please upload a CSV file before training')
    else:
        try:
            with st.spinner('Training models — this may take a minute...'):
                # Make load_and_train return both predictor and results
                predictor, results = load_and_train(uploaded_file)
                st.session_state["predictor"] = predictor
                st.session_state["last_results"] = results
                st.success('Training finished ✅')
        except Exception as e:
            st.sidebar.error(f'Failed to train: {e}')


           
                

# --- Show model results ---
st.header("Model Metrics")
results = st.session_state.get("last_results", predictor.model_results)
if results:
    for name, res in results.items():
        st.subheader(name)
        st.write(f"MSE: {res['mse']}, R²: {res['r2']}")
else:
    st.info("No trained models available. Upload a CSV and train.")

# --- Prediction Panel ---
st.header("Make a Prediction")
with st.form("predict_form"):
    feature_input = st.text_area(
        "Enter feature JSON (e.g. {'Slope Angle': 28, 'Rainfall': 10, 'Elevation': 1200})"
    )
    model_choice = st.selectbox("Select Model", options=list(results.keys()) if results else ["Random Forest"])
    submit_pred = st.form_submit_button("Predict")

if submit_pred:
    try:
        feature_data = json.loads(feature_input)
        pred_res = predictor.predict_with_explanation(feature_data, model_choice)
        st.json(pred_res)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Knowledge Base Query ---
st.header("Ask the Knowledge Base")
query_text = st.text_input("Enter your question (e.g. 'slope angle 30 risk')")
if st.button("Query Knowledge Base"):
    if query_text:
        answers = predictor.knowledge_base.query_knowledge(query_text)
        st.write({
            "question": query_text,
            "answers": answers,
            "timestamp": datetime.now().isoformat()
        })
    else:
        st.error("Please enter a question")
