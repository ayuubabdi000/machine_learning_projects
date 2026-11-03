import pandas as pd

CSV_path = './dataset/insurance.csv'
OUT_path = './dataset/clean_insurance.csv'

#load dataframe 

df = pd.read_csv(CSV_path)

# snapshot
print("\n=== INITIAL HEAD")

print(df.head())

print("\n=== INITIAL INFO")
print(df.info())


print("\n=== INITIAL Missing Value")
print(df.isnull().sum())

df = df.drop_duplicates()

# encode binary categoreis

df['sex'] = df['sex'].map({"female":0, "male":1 })

df['smoker'] = df['smoker'].map({"no":0,"yes":1})

# Feature engneering

df['bmi_smoker'] = df['bmi'] * df['smoker']
# One Hot Encoding
df =pd.get_dummies(df,columns=['region'],drop_first=True, dtype=int)

print(df.head)

print("\ ===al info===")
print(df.info)
print("\n ===final missing valus")
print(df.isnull().sum())

df.to_csv(OUT_path,index=False)

print(f"\nSaved to: {OUT_path}")
