import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# importing classification models
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TRAIN_PATH = "data/raw/train.csv"
SAVE_DIR = "artifacts/models/"

# Load the dataset
data = pd.read_csv(TRAIN_PATH)
# X = data.drop('rainfall', axis=1)
X = data.drop(columns=[ "mintemp", "dewpoint","pressure",'maxtemp','rainfall'])
y = data['rainfall']


# classification model map to explore
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'gradient_boosting': GradientBoostingClassifier(random_state=42),
    'extra_trees': ExtraTreesClassifier(random_state=42),
    'xgboost': XGBClassifier(random_state=42),
    'catboost': CatBoostClassifier(random_state=42, verbose=0),
    'lightgbm': LGBMClassifier(random_state=42),
    'svm': SVC(probability=True, random_state=42),
}


def train_model(model, X, y):
    ColumnTransformer([("power_transformer",PowerTransformer(method='yeo-johnson'),[])])
    # create a pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power_transform', PowerTransformer(method='box-cox')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X, y)

    # cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC AUC scores: {scores}")
    print(f"Mean ROC AUC: {scores.mean():.4f}, Std: {scores.std():.4f}")

    return scores.mean(), scores.std()

res = {}
for name, model in models.items():
    print("\n\n Model: ",name)
    mean, std = train_model(model,X,y)
    res[name] = {"mean":mean, "std":std}


res = pd.DataFrame(res).T

res.sort_values(by="mean",axis=0,ascending=False)

from sklearn.ensemble import VotingClassifier

# ensemble of extra_trees, catboost, lightgbm
# Create the individual models with probability outputs
extra_trees = ExtraTreesClassifier(random_state=42)
catboost = CatBoostClassifier(random_state=42, verbose=0)
lightgbm = LGBMClassifier(random_state=42)

# Create the ensemble model
ensemble = VotingClassifier(
    estimators=[
        ('extra_trees', models['extra_trees']),
        ('catboost', models['catboost']),
        ('lightgbm', models['lightgbm'])
    ],
    voting='soft'  # Use probability estimates for voting
)



# Train and evaluate the ensemble
print("\n\nEnsemble Model (Extra Trees + CatBoost + LightGBM):")
ensemble_mean, ensemble_std = train_model(ensemble, X, y)
res['ensemble'] = {"mean": ensemble_mean, "std": ensemble_std}

# Display final results
print("\nFinal Results:")
print(res.sort_values(by="mean", ascending=False))


# dump ensemble
import joblib
joblib.dump(ensemble,"artifacts/models/ensemble.pkl")