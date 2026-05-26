import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("Housing.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Dataset cleaned successfully")

# -----------------------------
# FEATURE SELECTION
# Reduced parameters
# -----------------------------
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(
    y_train.values.reshape(-1, 1)
).flatten()

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = SVR(kernel='rbf')

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# Convert back
y_pred_actual = scaler_y.inverse_transform(
    y_pred.reshape(-1, 1)
)

# -----------------------------
# EVALUATION
# -----------------------------
score = r2_score(
    y_test,
    y_pred_actual
)

print("R2 Score:", score)

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler_X, open("scaler_X.pkl", "wb"))
pickle.dump(scaler_y, open("scaler_y.pkl", "wb"))

print("Model saved successfully!")