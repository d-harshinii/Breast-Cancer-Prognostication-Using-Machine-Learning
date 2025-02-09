from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

# Home route to render the HTML pagepython -c "import sys; print(sys.executable)"

@app.route("/")
def home():
    return render_template("index.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        input_data = request.form["input_data"]
        
        # Convert input to NumPy array
        input_values = np.array([float(x) for x in input_data.split(",")])
        
        # Reshape and scale the input data
        input_data_reshaped = input_values.reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_reshaped)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        # Determine result
        result = "The Breast Cancer is Benign" if prediction[0] == 1 else "The Breast Cancer is Malignant"

        return render_template("index.html", prediction_result=result)

    except Exception as e:
        return render_template("index.html", prediction_result="Invalid input! Please enter 30 numeric values.")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
