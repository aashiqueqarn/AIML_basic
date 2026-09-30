import pandas as pd

data = {
    "name": ["Amit", "Priya", "Neha", "Sneha"],
    "age": [25, 30, 35, 40],
    "city": ["Patna", "Delhi", "Mumbai", "Patna"],
    "salary": [70000, 80000, 90000, 100000]
}

df = pd.DataFrame(data)
print(df.head())
print(df.info())
print(df.describe())
print(df["name"])
print(df[["name", "salary"]])
print("*"*100)
print(df[df["age"] > 30])
print("*"*100)
print(df[(df["city"] == "Patna") & (df["salary"] > 75000)])
print("*"*100)
above_avg = df[df["salary"] > df["salary"].mean()]
print(above_avg)
print("*"*100)
df_sorted = df.sort_values(by="age", ascending=False)
print(df_sorted)

