import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Menangani kebutuhan pelatihan khusus model
        if model_name == 'CatBoost':
            cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]
            model.fit(X_train, y_train, cat_features=cat_features_indices)
        elif model_name == 'LightGBM':
            X_train_lgb = X_train.copy()
            X_test_lgb = X_test.copy()
            for col in categorical_cols:
                if col in X_train.columns:
                    X_train_lgb[col] = X_train_lgb[col].astype('category')
                    X_test_lgb[col] = X_test_lgb[col].astype('category')
            model.fit(X_train_lgb, y_train)
            y_pred = model.predict(X_test_lgb)
            y_proba = model.predict_proba(X_test_lgb)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Menghitung metrik
        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        
        # Mencatat metrik ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc_roc)
        
        # Membuat dan menyimpan plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plot_path = f"plots/{model_name}_cm.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Membuat dan menyimpan plot kurva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plot_path = f"plots/{model_name}_roc.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        print(f"{model_name} - Accuracy: {acc:.4f}, AUC-ROC: {auc_roc:.4f}")

if __name__ == "__main__":
    # Untuk pengujian lokal dengan server MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Personality_Prediction")
    
    # Memuat dan mengacak data
    df = pd.read_csv('personality_dataset_preprocessing.csv')
    
    # Mendefinisikan variabel global untuk menyimpan nama-nama kolom bertipe kategorikal
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Membagi data
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Mendefinisikan model dengan hyperparameter tetap
    models = [
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
        ("CatBoost", CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=0)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    # Melatih dan mencatat setiap model
    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)