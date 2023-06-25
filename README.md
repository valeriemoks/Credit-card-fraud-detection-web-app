# Credit-card-fraud-detection-web-app
This is a web application for predicting credit card fraud using machine learning. The application takes in various features such as distance from home, distance from last transaction, ratio to median purchase price, and other factors to make a prediction whether a credit card transaction is fraudulent or not.

# Getting Started
To run the application, you will need to install Python 3 and the required libraries listed in requirements.txt. You can install the required libraries by running:

pip install -r requirements.txt
Once the libraries are installed, you can run the web application by executing:
python app.py

## Deployment
Use this start up command for deployment: `gunicorn -b 0.0.0.0:5000 app:app`

# Usage
To use the application, simply fill out the form on the web page with the required information and click "Predict". The application will then display the predicted result of whether the transaction is fraudulent or not.

# Model
The model used in the application is a logistic regression model trained on a dataset of credit card transactions. The model achieved an accuracy of over 95% on the test set.

# Acknowledgements
This application was built using Flask, a lightweight web application framework for Python. The dataset used for training the model was obtained from Kaggle.
