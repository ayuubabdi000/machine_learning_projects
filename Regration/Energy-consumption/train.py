import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATAPATH = './Dataset/clean_energy.csv'

df = pd.read_csv(DATAPATH)

# define feature and target

x = df.drop(
    columns=['EnergyConsumption']
)
y = df['EnergyConsumption']

x_train, x_test, y_train ,y_test = train_test_split(x,y, test_size=0.2,random_state=42)


colsscale=[
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "RenewableEnergy",
    "Hour"

]

scaler = StandardScaler()

x_train_lr =x_train.copy()
y_test_lr = y_test.copy()


x_train_lr[colsscale] = scaler.fit_transform(
    x_train[colsscale]
)
x_test[colsscale] = scaler.fit_transform(
    x_test[colsscale]
)

lr = LinearRegression()
lr.fit(
    x_train_lr,y_train
)


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(x_train,y_train)

os.makedirs("./models", exist_ok=True)

joblib.dump(
    lr,
    "./models/linear_regression.pkl"
)

joblib.dump(
    rf,
    "./models/random_forest.pkl"
)

joblib.dump(
    scaler,
    "./models/scaler.pkl"
)

joblib.dump(
    x.columns.tolist(),
    "./models/feature_columns.pkl"
)

print("Training complete.")
print("Saved Linear Regression, Random Forest, scaler, and feature columns.")

