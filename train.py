import pandas as pd
from pycaret.classification import *
import joblib

# Load dataset
data = pd.read_csv("data/iris.csv")

# Setup
clf = setup(
    data=data,
    target='Species',
    session_id=123,
    verbose=False
)

# Train best model
best_model = compare_models()

# Final model
final_model = finalize_model(best_model)

# Save pure sklearn model
joblib.dump(final_model, "models/iris_model.pkl")

print("Model Saved Successfully")