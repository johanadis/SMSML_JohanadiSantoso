import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, feature_names, params=None):
    with mlflow.start_run(run_name=model_name):
        start = time.time()
        if model_name == 'CatBoost Tuned':
            cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]
            model.fit(X_train, y_train, cat_features=cat_features_indices)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        elif model_name == 'LightGBM Tuned':
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
        
        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        training_time = end - start
        
        mlflow.log_param("model_type", model_name)
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("training_time", training_time)
        
        # Prepare input example for model signature
        input_example = X_train[:5]
        
        if model_name.startswith("XGBoost"):
            mlflow.xgboost.log_model(model, model_name, input_example=input_example)
        elif model_name.startswith("LightGBM"):
            mlflow.lightgbm.log_model(model, model_name, input_example=input_example)
        elif model_name.startswith("CatBoost"):
            mlflow.catboost.log_model(model, model_name, input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, model_name, input_example=input_example)
        
        # Plot confusion matrix
        plot_dir = "Membangun_model/Plots"
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
        
        # Plot ROC curve
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
        
        # Plot feature importance for tree-based models
        if model_name in ["Random Forest Tuned", "XGBoost Tuned", "LightGBM Tuned", "CatBoost Tuned"]:
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
    # Untuk pengujian lokal dengan server MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Menetapkan nama eksperimen MLflow: Personality_Prediction,  untuk mencatat seluruh run/model yang akan dilatih
    mlflow.set_experiment("Personality_Prediction")
    
    # Memuat dan mengacak data
    df = pd.read_csv('personality_dataset_preprocessing.csv')
    
    # Mendefinisikan variabel global untuk menyimpan nama-nama kolom bertipe kategorikal
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Split data
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Logistic Regression (baseline)
    lr_model = LogisticRegression(random_state=42)
    train_and_log_model(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test, feature_names)
    
    # Random Forest with GridSearchCV
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10]
    }
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    train_and_log_model(rf_grid.best_estimator_, "Random Forest Tuned", X_train, X_test, y_train, y_test, feature_names, rf_grid.best_params_)
    
    # XGBoost with GridSearchCV
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 10]
    }
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='accuracy')
    xgb_grid.fit(X_train, y_train)
    train_and_log_model(xgb_grid.best_estimator_, "XGBoost Tuned", X_train, X_test, y_train, y_test, feature_names, xgb_grid.best_params_)
    
    # LightGBM with GridSearchCV
    lgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 10]
    }
    lgb_model = LGBMClassifier(random_state=42, verbose=-1)
    lgb_grid = GridSearchCV(lgb_model, lgb_param_grid, cv=5, scoring='accuracy')
    lgb_grid.fit(X_train, y_train)
    train_and_log_model(lgb_grid.best_estimator_, "LightGBM Tuned", X_train, X_test, y_train, y_test, feature_names, lgb_grid.best_params_)
    
    # CatBoost with GridSearchCV
    cat_param_grid = {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'depth': [3, 5, 10]
    }
    cat_model = CatBoostClassifier(random_state=42, verbose=0)
    cat_grid = GridSearchCV(cat_model, cat_param_grid, cv=5, scoring='accuracy')
    cat_grid.fit(X_train, y_train)
    train_and_log_model(cat_grid.best_estimator_, "CatBoost Tuned", X_train, X_test, y_test, feature_names, cat_grid.best_params_)

if __name__ == "__main__":
    main()