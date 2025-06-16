import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set batas maksimum penggunaan CPU ke 4 core
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Aktifkan autologging MLflow untuk otomatis mencatat parameter, metrik, dan model
mlflow.autolog()
        
def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):

    # Mulai run MLflow untuk setiap model
    with mlflow.start_run(run_name=model_name):
        
        # Melatih model dengan data training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
        # Menghitung metrik akurasi dan AUC-ROC
        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        
        # Logging metrik ke MLflow
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
        # Mencatat confusion matrix sebagai CSV
        cm_csv_path = f"plots/{model_name}_cm.csv"
        pd.DataFrame(cm).to_csv(cm_csv_path, index=False)
        mlflow.log_artifact(cm_csv_path)
        
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
        # Mencatat data ROC sebagai CSV
        roc_csv_path = f"plots/{model_name}_roc.csv"
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(roc_csv_path, index=False)
        mlflow.log_artifact(roc_csv_path)
        
        print(f"{model_name} - Accuracy: {acc:.4f}, AUC-ROC: {auc_roc:.4f}")

def main():
    # Konfigurasi MLflow untuk tracking lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Menetapkan nama eksperimen MLflow: Personality_Prediction
    mlflow.set_experiment("Personality_Prediction")
    
    # Memuat dan mengacak data hasil preprocessing
    df = pd.read_csv('personality_dataset_preprocessing.csv')
    
    # Membagi data menjadi fitur (X) dan target (y)
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    
    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Mendefinisikan model dengan hyperparameter tetap
    models = [
        ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    # Melatih dan mencatat setiap model
    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)
        
if __name__ == "__main__":
    main()
