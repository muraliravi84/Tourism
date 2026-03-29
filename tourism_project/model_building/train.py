import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo

# HF API — token injected by GitHub Actions secret
api = HfApi()

# Load splits from HF Hub (available after prep.py runs in CI/CD)
Xtrain = pd.read_csv('hf://datasets/Murali0606/tourismdataset/Xtrain.csv')
Xtest  = pd.read_csv('hf://datasets/Murali0606/tourismdataset/Xtest.csv')
ytrain = pd.read_csv('hf://datasets/Murali0606/tourismdataset/ytrain.csv').squeeze()
ytest  = pd.read_csv('hf://datasets/Murali0606/tourismdataset/ytest.csv').squeeze()
print(f'Data loaded — Train: {Xtrain.shape}, Test: {Xtest.shape}')

# Class imbalance weight
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Pipeline: scale → XGBoost
model_pipeline = make_pipeline(
    StandardScaler(),
    xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators':     [50, 100],
    'xgbclassifier__max_depth':        [3, 4],
    'xgbclassifier__learning_rate':    [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.5, 0.6],
    'xgbclassifier__reg_lambda':       [0.4, 0.5]
}

# Train with GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

# Predictions with threshold
classification_threshold = 0.45
y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= classification_threshold).astype(int)

# Reports
train_report = classification_report(ytrain, y_pred_train)
test_report  = classification_report(ytest,  y_pred_test)
print('--- Train Classification Report ---')
print(train_report)
print('--- Test Classification Report ---')
print(test_report)

# Save model locally inside artifacts folder
artifact_dir = "tourism_project/artifacts"
os.makedirs(artifact_dir, exist_ok=True)
model_path = os.path.join(artifact_dir, "best-tourism-model-v1.joblib")
joblib.dump(best_model, model_path)
print(f"Model saved: {model_path}")

# Upload model to HF Hub
repo_id, repo_type = "Murali0606/tourism-model", "model"
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best-tourism-model-v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Model uploaded to HF Hub: {repo_id}")
