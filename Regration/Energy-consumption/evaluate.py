import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


DATAPATH = './Dataset/clean_energy.csv'

df = pd.read_csv(DATAPATH)


x = df.drop(
    columns=['EnergyConsumption']
)
y = df['EnergyConsumption']

x_train, x_test, y_train ,y_test = train_test_split(x,y, test_size=0.2,random_state=42)



lr = joblib.load(
    "./models/linear_regression.pkl"
)
rf = joblib.load(
    './models/random_forest.pkl'
)
scaler = joblib.load(
    './models/scaler.pkl'
)

feature_columns= joblib.load(
    './models/feature_columns.pkl'
)
scale_cols = [
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "RenewableEnergy",
    "Hour"
]

x_test_lr = x_test.copy()

x_test_lr[scale_cols] = scaler.transform(
    x_test[scale_cols]
)

# 4) Predictions
p_lr = lr.predict(x_test_lr)
p_rf = rf.predict(x_test)


def evaluate_models(name, y_true,y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true,y_pred)
    
    print(f"\n=== {name} ===")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")


    

evaluate_models(
    "Leanear Regration",
    y_test,
    p_lr
)

evaluate_models(
    "RANDOM FOREST",
    y_test,
    p_rf
)

i = 3

x_one_df = x_test.iloc[[i]]
y_true = float(y_test.iloc[i])


x_one_lr = x_one_df.copy()

x_one_lr[scale_cols] = scaler.transform(
    x_one_df[scale_cols]
)

p_lr_one = float(
    lr.predict(x_one_lr)[0]
)

p_rf_one = float(
    rf.predict(x_one_df)[0]
)

print("\n=== SINGLE-ROW SANITY CHECK ===")
print(f"Actual Energy: {y_true:,.2f}")
print(f"LR Prediction: {p_lr_one:,.2f}")
print(f"RF Prediction: {p_rf_one:,.2f}")
