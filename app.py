import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load saved model
with open("models/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Iris target names
species = ["setosa", "versicolor", "virginica"]


@app.route("/")
def home():
    return "Iris Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get input values
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        # Convert to numpy array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(features)[0]
        result = species[prediction]

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)