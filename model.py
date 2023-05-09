import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import pickle

df = pd.read_csv("credit card fraud data.csv")

# Get all the columns from the dataframe
columns = df.columns.tolist()

# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["fraud", "state", "category"]]

# Store the variable we are predicting 
target = "fraud"

# Convert the categorical columns to numerical columns
df["state"] = pd.factorize(df["state"])[0]
df["category"] = pd.factorize(df["category"])[0]

X = df[columns]
y = df[target]

model = LGBMClassifier(n_estimators=2000, 
                       learning_rate=0.01,
                       num_leaves=80,
                       colsample_bytree=0.98,
                       subsample=0.78,
                       reg_alpha=0.04,
                       reg_lambda=0.073,
                       subsample_for_bin=50,
                       boosting_type='gbdt',
                       is_unbalance=False,
                       min_split_gain=0.025,
                       min_child_weight=40,
                       min_child_samples=510,
                       objective='binary',
                       random_state=42,
                       n_jobs=-1)

model.fit(X, y)

# Save the model as a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
