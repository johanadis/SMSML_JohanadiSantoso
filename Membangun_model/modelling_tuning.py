import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import LabelEncoder
import joblib
# Set batas maksimum penggunaan CPU ke 4 core
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, feature_names, categorical_cols, params=None):
    # Mulai run MLflow untuk logging eksperimen
    with mlflow.start_run(run_name=model_name):
        start = time.time()
        # Penanganan fitur kategorikal khusus untuk LightGBM
        if isinstance(model, LGBMClassifier):
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
        
        end = time.time()
        
        # Hitung metrik evaluasi
        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        training_time = end - start
        
        # Logging parameter dan metrik ke MLflow
        mlflow.log_param("model_type", model_name)
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("training_time", training_time)
        
        input_example = X_train[:5]
        
        # Logging model ke MLflow
        if isinstance(model, LGBMClassifier):
            mlflow.lightgbm.log_model(model, model_name, input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)

        # Simpan model secara lokal dan log ke MLflow
        artifact_dir = "artifacts/tuned"
        os.makedirs(artifact_dir, exist_ok=True)
        if isinstance(model, LGBMClassifier):
            model_path = os.path.join(artifact_dir, f"{model_name}_model.txt")
            model.booster_.save_model(model_path)
        else:
            model_path = os.path.join(artifact_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Plot confusion matrix dan log ke MLflow
        plot_dir = "plots/tuned"
        os.makedirs(plot_dir, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(plot_dir, f"{model_name}_cm.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Plot ROC curve dan log ke MLflow
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        roc_path = os.path.join(plot_dir, f"{model_name}_roc.png")
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()
        
        # Plot feature importance jika tersedia
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
            plt.title(f'Feature Importance ({model_name})')
            plt.tight_layout()
            feat_imp_path = os.path.join(plot_dir, f"{model_name}_feature_importance.png")
            plt.savefig(feat_imp_path)
            mlflow.log_artifact(feat_imp_path)
            plt.close()
        
        print(f"{model_name} - Accuracy: {acc:.4f}, AUC-ROC: {auc_roc:.4f}, Training Time: {training_time:.4f}s")

def main():
    # Konfigurasi MLflow untuk tracking ke Dagshub
    tracking_uri = 'https://dagshub.com/johanadis/Eksperimen_SML_JohanadiSantoso.mlflow'
    username = 'johanadis' 
    token = os.getenv('DAGSHUB_TOKEN')
    
    if not token:
        raise ValueError("DAGSHUB_TOKEN not set in environment")
    
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Personality_Prediction")
    
    # Load dataset hasil preprocessing
    df = pd.read_csv('Membangun_model/personality_dataset_preprocessing.csv')
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Pisahkan fitur dan target
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    
    # Encode label target menjadi numerik
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    feature_names = X.columns.tolist()
    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10]
    }
    rf_model = RandomForestClassifier(random_state=42)
    # Hyperparameter tuning dengan GridSearchCV
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    train_and_log_model(rf_grid.best_estimator_, "Random Forest Tuned", X_train, X_test, y_train, y_test, feature_names, categorical_cols, rf_grid.best_params_)
    
    # LightGBM
    lgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 10]
    }
    lgb_model = LGBMClassifier(random_state=42, verbose=-1)
    # Hyperparameter tuning dengan GridSearchCV
    lgb_grid = GridSearchCV(lgb_model, lgb_param_grid, cv=5, scoring='accuracy')
    lgb_grid.fit(X_train, y_train)
    train_and_log_model(lgb_grid.best_estimator_, "LightGBM Tuned", X_train, X_test, y_train, y_test, feature_names, categorical_cols, lgb_grid.best_params_)

if __name__ == "__main__":
    main()