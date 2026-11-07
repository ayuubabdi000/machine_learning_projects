import joblib
import pandas as pd

price_per_kwh = 0.50
currency = "USD"

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
print(type(scaler))

# 2) New energy usage data
temperature = 25.0
humidity = 45.0
square_footage = 1500.0
occupancy = 5
hvac_usage = "On"
lighting_usage = "On"
renewable_energy = 15.0
day_of_week = "Monday"
holiday = "No"
hour = 14

hvac_value = 1 if hvac_usage == "On" else 0
lighting_value = 1 if lighting_usage == "On" else 0
holiday_value = 1 if holiday == "Yes" else 0

newdata = pd.DataFrame({
    'Temperature' : [temperature],
    'Humidity': [humidity],
    'SquareFootage':[square_footage],
    'Occupancy':[occupancy],
    'HVACUsage':[hvac_value],
    'LightingUsage':[lighting_value],
    'RenewableEnergy':[renewable_energy],
    'Holiday':[holiday_value],
    'Hour':[hour],
    f"DayOfWeek_{day_of_week}":[1]

})

newdata = newdata.reindex(
    columns=feature_columns,
    fill_value=0
)


scale_cols=[
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "RenewableEnergy",
    "Hour"
]

newdata_lr = newdata.copy()
newdata_lr[scale_cols] = scaler.transform(
    newdata[scale_cols]
)

lr_prediction = float(
    lr.predict(newdata_lr)[0]
)
print(f"Lewanier regration: {lr_prediction}")
rf_predition = float(
    rf.predict(newdata)[0]
)
print(f"Rondom Forest: {rf_predition}")