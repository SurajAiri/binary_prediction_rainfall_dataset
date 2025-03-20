import pandas as pd
import joblib
import os

DATA_PATH = "data/raw/test.csv"
MODEL_PATH = "artifacts/models/logistic_regression.pkl"
SAVE_DIR = "artifacts/predictions/"
SAVE_PATH = os.path.join(SAVE_DIR,"prediction.csv")

# Make sure the artifacts directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # Load the dataset
    data = pd.read_csv(DATA_PATH, index_col=0)

    # Load the model
    model = joblib.load(MODEL_PATH)

    # Make predictions
    y_pred = model.predict_proba(data)

    # Save the predictions
    pd.DataFrame(y_pred[:,1], columns=["rainfall"], index=data.index).to_csv(SAVE_PATH)

    print("Predictions saved to", SAVE_PATH)

if __name__ == "__main__":
    main()