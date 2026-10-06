import os
import shutil
import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), "result_image")
os.makedirs(RESULT_DIR, exist_ok=True)
dataset_dir = os.path.join(os.getcwd(), "dataset")
os.makedirs(dataset_dir, exist_ok=True)

dataset_path = kagglehub.dataset_download("yasserh/titanic-dataset")

target_dir = os.path.join(dataset_dir, "titanic-dataset")
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.move(dataset_path, target_dir)

print("Dataset saved to:", target_dir)

dataset = pd.read_csv(os.path.join(target_dir, "Titanic-Dataset.csv"))
print("Dataset info", dataset.info())
print("Describe dataset", dataset.describe())
print("Missing values per column:\n", dataset.isna().sum())
print("Null values per column:\n", dataset.isnull().sum())
print("Missing values percentage:\n", dataset.isna().mean().mul(100).round(2))

for col in dataset.select_dtypes(include="number").columns:
    if dataset[col].isna().any():
        dataset[col].fillna(dataset[col].median(), inplace=True)

for col in dataset.select_dtypes(include="object").columns:
    if dataset[col].isna().any():
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

sns.boxplot(data=dataset, x="Fare")
plt.title("Fare Distribution")
plt.savefig(os.path.join(RESULT_DIR, "fare_distribution.png"))

plt.close('all')  # clear previous figures/axes
selected_cols = ["Age", "Fare"]
for i, col in enumerate(selected_cols, 1):
    plt.subplot(1, 2, i)
    dataset[col].dropna().hist(bins=20, edgecolor='black')
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"age_fare_distribution.png"))
plt.close('all')

dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
plt.figure(figsize=(8,6))
sns.scatterplot(data=dataset, x="Fare", y="Age", alpha=0.7)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "age_fare_scatterplot.png"))

avg_fare_by_class = (
    dataset.groupby("Pclass", as_index=False)["Fare"]
    .mean()
    .rename(columns={"Fare": "avg_fare"})
    .sort_values("Pclass")
)
plt.figure(figsize=(8,6))
sns.barplot(data=avg_fare_by_class, x="Pclass", y="avg_fare", palette="viridis")
plt.title("Average Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "avg_fare_by_class.png"))

plt.close('all')
numeric_df = dataset.select_dtypes(include="number")
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Heatmap for all numeric columns")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "correlation_heatmap_of_titanic_dataset.png"))

