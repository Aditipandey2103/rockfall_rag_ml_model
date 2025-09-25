# RAG-Enhanced Rockfall Prediction System (Flask)
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Optional RAG imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG not available, running in basic mode")

# ---------------- RAG Knowledge Base ----------------
class RockfallKnowledgeBase:
    def __init__(self):
        self.knowledge_base = self._create_basic_knowledge()
        self.embeddings_model = None
        self.vector_db = None
        self.collection = None
        if RAG_AVAILABLE:
            try:
                self._initialize_rag_components()
            except Exception as e:
                print(f"RAG init failed: {e}")
    
    def _create_basic_knowledge(self):
        return {
            'slope_thresholds': {
                'low_risk': {'range': (0, 15), 'description': 'Stable slopes, minimal rockfall risk'},
                'medium_risk': {'range': (15, 25), 'description': 'Moderate risk, monitor during rainfall'},
                'high_risk': {'range': (25, 35), 'description': 'High risk, active monitoring needed'},
                'critical_risk': {'range': (35, 90), 'description': 'Critical risk, immediate intervention'}
            },
            'environmental_triggers': {
                'rainfall': 'Heavy rainfall (>25mm/hr) is a primary rockfall trigger',
                'temperature': 'Freeze-thaw cycles weaken rock over time'
            }
        }
    def _initialize_rag_components(self):
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = chromadb.Client()
        try:
            self.collection = self.vector_db.get_collection(name="rockfall_knowledge")
        except Exception:
            self.collection = self.vector_db.create_collection(name="rockfall_knowledge")
        self._populate_vector_db()   
    def _populate_vector_db(self):
        if not RAG_AVAILABLE or self.collection is None:
            return
        documents, metadatas, ids = [], [], []
        doc_id = 0
        for category, items in self.knowledge_base.items():
            for key, value in items.items():
                doc_text = value['description'] if isinstance(value, dict) else str(value)
                documents.append(f"{category} - {key}: {doc_text}")
                metadatas.append({'category': category, 'subcategory': key})
                ids.append(f"doc_{doc_id}")
                doc_id += 1
        try:
            embeddings = self.embeddings_model.encode(documents)
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Failed to populate RAG DB: {e}")
    
    def query_knowledge(self, query, n_results=3):
        if RAG_AVAILABLE and self.collection:
            try:
                query_emb = self.embeddings_model.encode([query])
                res = self.collection.query(query_embeddings=query_emb.tolist(), n_results=n_results)
                relevant = []
                for i, doc in enumerate(res['documents'][0]):
                    relevant.append({
                        'content': doc,
                        'category': res['metadatas'][0][i]['category'],
                        'subcategory': res['metadatas'][0][i]['subcategory'],
                        'relevance_score': 1.0 - res['distances'][0][i] if 'distances' in res else 0.8
                    })
                return relevant
            except Exception:
                return self._basic_query(query)
        else:
            return self._basic_query(query)
    
    def _basic_query(self, query):
        q = query.lower()
        relevant_info = []
        keywords_map = {
            'slope': 'slope_thresholds', 'rain': 'environmental_triggers', 'rainfall': 'environmental_triggers'
        }
        for k, cat in keywords_map.items():
            if k in q and cat in self.knowledge_base:
                relevant_info.append({'content': str(self.knowledge_base[cat]), 'category': cat, 'relevance_score': 0.7})
        if not relevant_info:
            relevant_info.append({'content': 'Monitor slope angle, rainfall, and environmental conditions', 'category':'general','relevance_score':0.5})
        return relevant_info
    
    # --- helper function ---
def generate_risk(df):
    slope_norm = df['Slope Angle'] / df['Slope Angle'].max()
    rainfall_norm = df['Rainfall'] / df['Rainfall'].max()
    depth_norm = df['Pitmine Depth'] / df['Pitmine Depth'].max()
    radius_norm = df['Pitmine Radius'] / df['Pitmine Radius'].max()
    elevation_norm = df['Elevation'] / df['Elevation'].max()
    
    risk_score = (
            slope_norm * 0.4 +
            rainfall_norm * 0.3 +
            depth_norm * 0.1 +
            radius_norm * 0.1 +
            elevation_norm * 0.1
        ) * 100
    
    df['Risk'] = risk_score
    return df
    
    
# ---------------- Predictor ----------------
class EnhancedRockfallPredictor:
    def __init__(self):
        self.models = {}
        self.model_results = {}
        self.knowledge_base = RockfallKnowledgeBase()
        self.label_encoders = {}

    def prepare_data(self, df):
        df = df.dropna(how='all')
        categorical_cols = ['Geography','Rainfall Pattern','Soil Type','Livelihood','Industrial Activity']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        if 'Risk' not in df.columns:
            raise ValueError("CSV must contain 'Risk' column")
        X = df.drop(columns=['Risk'])
        y = df['Risk']
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def train_models(self, X_train, y_train, X_test, y_test):
        self.feature_columns = X_train.columns.tolist()
        regression_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'SVM': SVR()
        }
        results = {}
        for name, model in regression_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'model': model,
                'mse': round(mean_squared_error(y_test, y_pred),4),
                'r2': round(r2_score(y_test, y_pred),4),
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist()
            }
        self.models = regression_models
        self.model_results = results
        return results
    
    def predict_with_explanation(self, input_features, model_name='Random Forest'):
        if model_name not in self.models:
            return {"error": f"Model {model_name} not trained"}
        # Make sure we know the feature order used during training
        if not hasattr(self, 'feature_columns'):
            return {"error": "Feature columns not set. Train the model first."}

        # Encode categorical fields using the same encoders from training
        encoded_features = {}
        for key in self.feature_columns:
            if key in input_features:
                val = input_features[key]
                if key in self.label_encoders:
                    le = self.label_encoders[key]
                    try:
                        encoded_features[key] = le.transform([val])[0]
                    except ValueError:
                    # unseen category → default 0
                        encoded_features[key] = 0
                else:
                    encoded_features[key] = val
            else:
            # missing feature → default 0
                encoded_features[key] = 0

        X = np.array([encoded_features[col] for col in self.feature_columns]).reshape(1, -1)
        
        

        
        
        model = self.models[model_name]
       

        
        pred = model.predict(X)[0]
        explanation = self.knowledge_base.query_knowledge(f"slope angle {input_features.get('Slope Angle',0)} rainfall {input_features.get('Rainfall',0)}")
        return {
            'prediction': round(float(pred),2),
            'model': model_name,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }

# ---------------- Flask App ----------------
# ---------------- Predictor Instance ----------------
predictor = EnhancedRockfallPredictor()  # <-- create the predictor object here
results = {}              # to store trained model results
risk_vs_features = {}     # to store risk vs feature data



# ---------------- Data Loading ----------------
def load_and_train(csv_path):
    global results, risk_vs_features

    df = pd.read_csv(csv_path)
    df = df.dropna(how='all')

    if 'Risk' not in df.columns:
        df = generate_risk(df)

    predictor = EnhancedRockfallPredictor()   # create fresh instance here
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)

    predictor.feature_columns = X_train.columns.tolist()
    results = predictor.train_models(X_train, y_train, X_test, y_test)

    # Risk vs features for analysis
    risk_vs_features = {}
    for col in X_train.columns:
        grouped = df.groupby(col)['Risk'].mean().reset_index()
        risk_vs_features[col] = {
            'x': grouped[col].tolist(),
            'y': grouped['Risk'].tolist()
        }

    print("Training complete!")
    print("Trained models:", list(predictor.models.keys()))

    return predictor, results

 
# ---------------- Run ----------------