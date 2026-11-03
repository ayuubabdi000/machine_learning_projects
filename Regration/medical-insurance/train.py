import os 
import joblib
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# DATA_PATH = './dataset/clean_insurance.csv'

# MODEL_PATH ='models/leanier_regration_model.pk1'
DATA_PATH = './dataset/clean_insurance.csv'
MODEL_PATH = 'models/linear_regression_model.pkl'

df = pd.read_csv(DATA_PATH)

# Define Feutures and target
x = df.drop(columns="charges")
y = df['charges']

x_train,x_test, y_train ,y_test =train_test_split(x,y, test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

