import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    distance_from_home = float(request.form['distance_from_home'])
    distance_from_last_transaction = float(request.form['distance_from_last_transaction'])
    ratio_to_median_purchase_price = float(request.form['ratio_to_median_purchase_price'])
    repeat_retailer = int(request.form['repeat_retailer'])
    used_chip = int(request.form['used_chip'])
    used_pin_number = int(request.form['used_pin_number'])
    online_order = int(request.form['online_order'])

    # Create a data array in the format that the model expects
    data = np.array([[distance_from_home,
                      distance_from_last_transaction,
                      ratio_to_median_purchase_price,
                      repeat_retailer,
                      used_chip,
                      used_pin_number,
                      online_order]])

    # Make a prediction using the model
    prediction = model.predict(data)

    # Return the prediction
    if prediction[0] == 1:
        return render_template('index.html', prediction_text='The prediction is: Fraudulent Transaction')
    else:
        return render_template('index.html', prediction_text='The prediction is: Legitimate Transaction')

if __name__ == '__main__':
    app.run(debug=True)
