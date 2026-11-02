import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "age": [25, np.nan, 35, 40, np.nan, 22],
    "income": [50000, 60000, np.nan, 80000, 45000, np.nan],
    "city": ["Patna", "Delhi", "Patna", np.nan, "Mumbai", "Delhi"]
})


print("Sum", df.isnull().sum())

df["age"] = df["age"].fillna(df["age"].median())
df["income"] = df["income"].fillna(df["income"].median())
df["city"] = df["city"].fillna(df["city"].mode()[0])

print(df)


df1 = pd.DataFrame({
    "age": [25, np.nan, 35, 40, np.nan, 22],
    "income": [50000, 60000, np.nan, 80000, 45000, np.nan],
    "city": ["Patna", "Delhi", "Patna", np.nan, "Mumbai", "Delhi"]
})
imputer = SimpleImputer(strategy="most_frequent")
df1[["age", "income"]] = imputer.fit_transform(df1[["age", "income"]])
print(df1)


