import pandas as pd


#load dataset

DATA_PATH = './Dataset/Energy_consumption.csv'
df = pd.read_csv(DATA_PATH)
# snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())
print("\n=== INITIAL info ===")
print(df.info())
print("\n=== INITIAL missing values ===")
print(df.isnull().sum())

df = df.drop_duplicates()



df['Timestamp'] = pd.to_datetime(df['Timestamp'])

#encode 
df['LightingUsage'] = df['LightingUsage'].map({
    'On':1,
    'Off':0
})

df['Holiday'] = df['Holiday'].map({
    'Yes':1,
    'No':0 
})
df['HVACUsage'] = df['HVACUsage'].map({
    'On':1,
    'Off':0
})

#feature engneering
df["Hour"] = df["Timestamp"].dt.hour
df.drop('Timestamp', axis=1, inplace=True)
df = pd.get_dummies(df,columns=['DayOfWeek'],dtype=int)

OUTPATH = './Dataset/clean_energy.csv'
df.to_csv(OUTPATH,index=False)

print(f"\n saved to {OUTPATH}")