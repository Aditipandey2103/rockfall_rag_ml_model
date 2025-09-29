import streamlit as st
import pandas as pd
import json
from datetime import datetime
import numpy as np

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
    
    model_options = list(results.keys()) if results else ["Random Forest"]
    model_options.append("All Models")
    model_choice = st.selectbox("Select Model", options=model_options)

    submit_pred = st.form_submit_button("Predict")


if submit_pred:
    try:
        feature_data = json.loads(feature_input)
        
        # new code
        if model_choice == "All Models":
            preds = {}
            for name in results.keys():
                preds[name] = predictor.predict_with_explanation(feature_data, name)

    # Calculate ensemble average
            avg_score = np.mean([p["prediction"] for p in preds.values()])

    # Map to user-friendly risk label
            # Map to user-friendly risk label, immediate actions, and consequences
            if avg_score < 0.3:
                risk_label = "Low Risk ✅ (Site appears safe)"
                actions = [
                "Routine monitoring of slopes",
                "Ensure standard safety protocols are followed"
                ]
                consequences = "Minimal risk of rockfall; operations can continue normally."
            elif avg_score < 0.6:
                risk_label = "Moderate Risk ⚠️ (Monitor conditions closely)"
                actions = [
            "Increase frequency of slope inspections",
            "Restrict access to high-risk areas during rainfall",
            "Prepare emergency response team"
                ]
                consequences = "Possible minor rockfall; caution required for personnel."
            else:
                risk_label = "High Risk 🚨 (Unsafe, potential rockfall)"
                actions = [
            "Evacuate personnel from the site immediately",
            "Suspend all ongoing operations",
            "Deploy emergency response teams",
            "Issue alert to local authorities"
                ]
                consequences = "High probability of rockfall; severe damage or casualties possible."

# Show clean summary
            st.subheader("Prediction Summary")
            st.metric("Risk Score", f"{avg_score:.2f}")
            if avg_score < 0.3:
                st.success(risk_label)
            elif avg_score < 0.6:
                st.warning(risk_label)
            else:
                st.error(risk_label)

# Show recommended actions and consequences
            st.subheader("Recommended Immediate Actions")
            for act in actions:
                st.write(f"- {act}")

            st.subheader("Potential Consequences")
            st.write(consequences)


   

    # (Optional) allow user to expand and see model-wise details
            with st.expander("See model-wise details"):
                df = pd.DataFrame([
                    {"Model": name, "Prediction": p["prediction"]}
                    for name, p in preds.items()
                    ])
                st.dataframe(df)
                st.json(preds)
        else:
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
