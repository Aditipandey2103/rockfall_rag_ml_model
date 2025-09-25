RAG-Enhanced Rockfall Prediction System

A Rockfall Prediction System enhanced with Retrieval-Augmented Generation (RAG) to predict landslide/rockfall risk based on environmental and geological factors. Built with Python, scikit-learn, and Streamlit, with optional RAG-based knowledge retrieval for contextual explanations.

Features

Predict Rockfall Risk: Uses Random Forest, Gradient Boosting, Linear Regression, and SVM models.

RAG Knowledge Base: Provides contextual explanations based on slope thresholds, rainfall patterns, and environmental triggers.

Custom Input: Enter features as JSON for predictions.

Auto Risk Calculation: Generates a Risk column if missing in the dataset.

Streamlit UI: Intuitive interface for training models, making predictions, and querying knowledge.

Dataset

The system expects a CSV with at least the following columns:

Slope Angle

Rainfall

Elevation

Geography

Rainfall Pattern

Soil Type

Livelihood

Industrial Activity

Pitmine Depth

Pitmine Radius

Risk (optional — can be auto-generated)
