import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(__file__), "result_image")
os.makedirs(RESULT_DIR, exist_ok=True)

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig(os.path.join(RESULT_DIR, "sin_wave.png"))

np.random.seed(0)
df = pd.DataFrame({
    'age': np.random.normal(35, 10, 200).astype(int),
    'income': np.random.normal(50000, 15000, 200)
})

sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.savefig(os.path.join(RESULT_DIR, "age_distribution.png"))

sns.scatterplot(data=df, x="age", y="income")
plt.title("Age vs Income")
plt.savefig(os.path.join(RESULT_DIR, "age_income_scatter.png"))
plt.close('all')

df = pd.DataFrame(np.random.rand(100, 4), columns=["A", "B", "C", "D"])
df["E"] = df["A"] * 2 + np.random.normal(0, 0.1, 100)
corr = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(RESULT_DIR, "correlation_heatmap.png"))


sales = pd.DataFrame({
    "city": ["Patna", "Delhi", "Patna", "Mumbai", "Delhi", "Mumbai"],
    "product": ["A", "B", "A", "A", "B", "B"],
    "amount": [100, 200, 150, 300, 250, 175]
})
# Aggregate total amount per city and plot
city_totals = sales.groupby('city', as_index=False)['amount'].sum().sort_values('amount', ascending=False)
plt.close('all')  # clear previous figures/axes
plt.figure(figsize=(8,6))
sns.barplot(data=city_totals, x='city', y='amount', order=city_totals['city'])
plt.title("Total sales amount per city")
plt.ylabel("Total amount")
plt.xlabel("city")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "total_amount_per_city.png"))