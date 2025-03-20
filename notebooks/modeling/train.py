import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
import os

import matplotlib.pyplot as plt

# Path constants
DATA_PATH = 'data/raw/train.csv'
MODEL_DIR = 'artifacts/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
ROC_CURVE_PATH = 'artifacts/roc_curve.png'

# Make sure the artifacts directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Main function using cross validation
def main_cv():
    # Load the dataset
    data = pd.read_csv(DATA_PATH, index_col=0)

    # data.drop(["temparature", "mintemp", "dewpoint","pressure","humidity"], axis=1, inplace=True)

    X = data.drop('rainfall', axis=1)
    y = data['rainfall']

    print(f"Dataset shape: {data.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")

    # Define the pipeline with preprocessing and model steps
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42))
    ])

    # Get predicted probabilities using cross-validation
    print("Performing 5-fold cross-validation...")
    y_probs = cross_val_predict(pipeline, X, y, cv=5, method='predict_proba')
    
    # Calculate traditional accuracy scores too
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print('Cross Validation Scores: ', scores)
    print('Mean Accuracy: ', scores.mean())
    print('Standard Deviation: ', scores.std())
    
    # Train the final model on the full dataset
    print("Training final model on full dataset...")
    pipeline.fit(X, y)
    
    # Save the model
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the ROC curve plot
    plt.savefig(ROC_CURVE_PATH)
    plt.show()
    
    # Calculate and print AUC score
    auc_score = roc_auc_score(y, y_probs[:, 1])
    print(f'ROC AUC Score: {auc_score:.3f}')
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal threshold: {optimal_threshold:.3f}')

if __name__ == '__main__':
    main_cv()