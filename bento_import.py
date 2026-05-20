import bentoml
from pycaret.classification import load_model

# Load trained PyCaret model
model = load_model("models/iris_model")

# Save model into BentoML model store
bentoml.sklearn.save_model(
    "iris_classifier",
    model
)

print("Model saved successfully in BentoML")