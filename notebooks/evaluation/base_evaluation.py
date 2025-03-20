import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(y_true, y_pred,save_results=True):
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    
    # Print metrics
    print('Accuracy: ', acc)
    print('F1 Score: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Confusion Matrix: \n', cm)
    print('Classification Report: \n', cr)
    
    if save_results:
        # Store metrics in a string
        eval_results = f"""
# Model Evaluation Results
Model: Random Forest Classifier

## Performance Metrics
- Accuracy: {acc}
- F1 Score: {f1}
- Precision: {precision}
- Recall: {recall}

## Confusion Matrix
{cm}

## Classification Report
{cr}
"""
        
        # Save the evaluation results to a file
        with open('evaluation_results.md', 'w') as file:
            file.write(eval_results)
    
    return acc, f1, precision, recall, cm, cr


def main():
    # Load the dataset
    data = pd.read_csv('data/raw/train.csv')
    X = data.drop('rainfall', axis=1)
    y = data['rainfall']

    # train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # evaluate the model
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)


# using cross validation
def main_cv():
    # Load the dataset
    data = pd.read_csv('data/raw/train.csv')

    # data.drop(["temparature", "mintemp", "dewpoint","pressure","humidity"], axis=1, inplace=True)

    X = data.drop('rainfall', axis=1)
    y = data['rainfall']

    # Create a pipeline with scaling and model training steps
    from sklearn.preprocessing import StandardScaler
    # from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline

    # Define the pipeline with preprocessing and model steps
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier())
    ])

    # Cross validation with pipeline
    from sklearn.model_selection import cross_val_predict, cross_val_score
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    import numpy as np

    # Get predicted probabilities using cross-validation
    y_probs = cross_val_predict(pipeline, X, y, cv=5, method='predict_proba')

    
    # # Calculate traditional accuracy scores too
    # scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    # print('Cross Validation Scores: ', scores)
    # print('Mean Accuracy: ', scores.mean())
    # print('Standard Deviation: ', scores.std())
    
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
    plt.savefig('roc_curve.png')
    plt.show()
    
    # Calculate and print AUC score
    auc_score = roc_auc_score(y, y_probs[:, 1])
    print(f'ROC AUC Score: {auc_score:.3f}')

if __name__ == '__main__':
    main_cv()