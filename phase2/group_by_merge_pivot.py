import pandas as pd

sales = pd.DataFrame({
    "city": ["Patna", "Delhi", "Patna", "Mumbai", "Delhi", "Mumbai"],
    "product": ["A", "B", "A", "A", "B", "B"],
    "amount": [100, 200, 150, 300, 250, 175]
})
print(sales.groupby("city")["amount"].sum())
print(sales.groupby(["city", "product"])["amount"].mean())
print("="*25)
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "name": ["Amit", "Priya", "Rahul"]
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 1, 3],
    "amount": [500, 300, 700, 200]
})

merged = pd.merge(customers, orders, on="customer_id")
print(merged)

print(merged.groupby("name")["amount"].sum())
print("="*25)

print(sales.pivot_table(index="city", columns="product", values="amount", aggfunc="sum"))
print("="*25)
transactions = pd.DataFrame({
    "date": [9, 10, 14, 12],
    "customer_id": [1, 2, 1, 3],
    "amount": [500, 300, 700, 200]
})
customer_feature = (
    transactions.groupby("customer_id")
    .agg(
        total_spend = ("amount", "sum"),
        avg_spend = ("amount", "mean"),
        transaction_count = ("amount", "count")
    )
    .reset_index()
)
print(customer_feature)