import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_model():
    """Train the anomaly detection model and save all components"""
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("smart_system_anomaly_dataset (1).csv")
    
    # Show dataset info
    print(f"Dataset shape: {df.shape}")
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    le = LabelEncoder()
    df["device_type"] = le.fit_transform(df["device_type"])
    
    # Features and labels
    X = df.drop(columns=["label", "timestamp", "device_id","timestamp","geo_location_variation"])
    le1 = LabelEncoder()
    df["label"] = le1.fit_transform(df["label"])
    y = df["label"]
    
    print(f"Features shape: {X.shape}")
    
    # Apply SMOTE for balancing
    print("\nApplying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Resampled dataset shape: {X_res.shape}, Class distribution: {np.bincount(y_res)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Train-test split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)
    
    # Handle binary vs multiclass ROC AUC
    if y_proba.shape[1] == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save all components
    print("\nSaving model components...")
    joblib.dump(xgb, 'models/xgb_anomaly_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl') 
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(le1, 'models/label_encoder1.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    # Save label mapping from encoder
    label_mapping = dict(zip(le1.transform(le1.classes_), le1.classes_))
    joblib.dump(label_mapping, 'models/label_mapping.pkl')
    
    print("Model training completed and saved successfully!")
    print("\nSaved files:")
    for f in [
        "models/xgb_anomaly_model.pkl",
        "models/scaler.pkl",
        "models/label_encoder.pkl",
        "models/label_encoder1.pkl",
        "models/feature_columns.pkl",
        "models/feature_importance.csv",
        "models/label_mapping.pkl",
    ]:
        print("-", f)

if __name__ == "__main__":
    train_and_save_model()
