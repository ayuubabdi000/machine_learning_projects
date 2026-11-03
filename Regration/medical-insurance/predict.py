import joblib
import pandas as pd
MODEL_PATH = './models/linear_regression_model.pkl'
model = joblib.load(MODEL_PATH)
age =30
sex = 'male'
bmi = 23.5
children = 0
smoker = 'no'
region = 'southwest'

sex_value =1 if sex == 'male' else 0
smoker_value = 1 if smoker == 'yes' else 0

new_data = pd.DataFrame({
    'age':[age],
    'sex':[sex_value],
    'bmi':[bmi],
    'children':[children],
    'smoker':[smoker_value],
    'bmi_smoker' : [bmi * smoker_value],
    'region_northwest':[1 if region == 'region_northwest' else 0],
    'region_southeast':[1 if region == 'region_southeast' else 0],
    'region_southwest':[1 if region =='region_southwest'else 0]
})

prediction = model.predict(new_data)[0]

print(f"predicted insurance cost:{prediction:,.2f}")