import os 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = './dataset/clean_insurance.csv'
MODEL_PATH = 'models/linear_regression_model.pkl'


# load data and model

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

x = df.drop(columns="charges")
y = df['charges']

x_train,x_test, y_train ,y_test =train_test_split(x,y, test_size=0.2,random_state=42)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("\n=== MODEL EVALUATION ===")
print(f"MAE: {mae: .2f}")
print(f"MSE: {rmse: .2f}")
print(f"R²: {r2: .4f}")


# 6
i = 3 
x_one_df =x_test.iloc[[i]]
y_true = y_test.iloc[i]

p_one = float(model.predict(x_one_df)[0])

print("\n=== single sanity check ===")
print(f" actually predicted : {y_true}")
print("LR Prediction: ${p_one:, .2f}")

os.makedirs('models',exist_ok=True)
joblib.dump(model,MODEL_PATH)

print(f"model saved to {MODEL_PATH}")
print(f"training smples{x_train}")
print(f"testing samples{x_test}")


os.makedirs("./outputs", exist_ok=True)
plt.scatter(y_test,y_pred, alpha=0.6)
plt.xlabel('Actual charges')
plt.ylabel('predicted charges')
plt.title('actual vs predicted')
plt.savefig('./outputs/actual_vs_predicted.png', bbox_inches="tight")
plt.close()

residual = y_test - y_pred
plt.scatter(y_test,residual, alpha=0.6)
plt.axhline(0)
plt.xlabel('predicted charges')
plt.ylabel('residual plot')
plt.title('residual plot')
plt.savefig('./outputs/residual.png', bbox_inches="tight")
print(' saved ')