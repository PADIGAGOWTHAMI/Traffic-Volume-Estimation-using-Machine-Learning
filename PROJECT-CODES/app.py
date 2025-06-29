import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template
import os

app = Flask(__name__, template_folder='template')

# Load model and encoder using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")

# 🔐 Safe loading
try:
    model = pickle.load(open(model_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or encoder: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
#@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get user input from HTML form
        input_feature = [x for x in request.form.values()]
        print("💻 RECEIVED INPUTS:", input_feature)
        print("🧠 FORM KEYS:", list(request.form.keys()))

        # Must receive exactly 12 values
        if len(input_feature) != 11:
            return render_template("output.html", result=f"❌ Only received {len(input_feature)} values. Please fill all 11 fields.")

        # Define input features
        original_features = ["holiday", "temp", "rain", "snow", "weather",
                             "year", "month", "day", "hours", "minutes", "seconds"]

        # Create DataFrame
        feature_values = [np.array(input_feature)]
        data = pd.DataFrame(feature_values, columns=original_features)

        # Convert numeric columns to float
        numeric_columns = ["temp", "rain", "snow", "year", "month", "day", "hours", "minutes", "seconds"]
        data[numeric_columns] = data[numeric_columns].astype(float)

        # Encode categorical values
        categorical_features = ["holiday", "weather"]
        encoded = encoder.transform(data[categorical_features])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))

        # Merge encoded with numeric
        data = data.drop(columns=categorical_features).reset_index(drop=True)
        data = pd.concat([data, encoded_df], axis=1)

        # Ensure column match with trained model
        data = data.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict
        prediction = model.predict(data)[0]
        result = f"🚦 Estimated Traffic Volume: {int(prediction):,}"

        return render_template("output.html", result=result)

    except Exception as e:
        return render_template("output.html", result=f"❌ Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
