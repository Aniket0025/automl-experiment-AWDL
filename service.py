import bentoml
import pandas as pd
import joblib
from bentoml.io import JSON

# Load model
model = joblib.load("models/iris_model.pkl")

# Create service
svc = bentoml.Service("iris_service")

@svc.api(input=JSON(), output=JSON())
def predict(input_data):

    try:

        # Extract features
        features = input_data["features"]

        # Convert to dataframe
        df = pd.DataFrame(
            [features],
            columns=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )

        # Predict
        prediction = model.predict(df)

        return {
            "prediction": prediction.tolist()
        }

    except Exception as e:

        return {
            "error": str(e)
        }