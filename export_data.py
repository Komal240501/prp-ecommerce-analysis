"""
Run this ONCE on your own PC (where SQL Server + the ODBC driver work).
Exports the six tables to ./data as Parquet (snappy-compressed) instead of
CSV. Parquet is typically 5-10x smaller than CSV for the same data, which
usually resolves GitHub's file-size limits (25MB via web upload, 100MB hard
cap via git push) for the large event-level tables (website_sessions,
website_pageviews).

Copy the resulting 'data' folder into your Streamlit deployment repo.
Requires: pip install pyarrow
"""
import os
import pandas as pd
import pyodbc

conn = pyodbc.connect(
    'Driver={SQL Server};'
    'Server=KOMAL\\SQLEXPRESS;'
    'Database=Ecommerce_Analytics_project;'
    'Trusted_connection=yes;'
)

os.makedirs("data", exist_ok=True)

tables = {
    "sessions": "SELECT * FROM website_sessions",
    "orders": "SELECT * FROM orders",
    "order_items": "SELECT * FROM orders_items",
    "products": "SELECT * FROM products",
    "refunds": "SELECT * FROM order_item_refunds",
    "website_pageviews": "SELECT * FROM website_pageviews",
}

MAX_MB_BEFORE_WARN = 25  # GitHub's web-upload limit

for name, query in tables.items():
    df = pd.read_sql(query, conn)
    path = f"data/{name}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    flag = "  <-- still large, consider aggregating or Git LFS" if size_mb > MAX_MB_BEFORE_WARN else ""
    print(f"Saved {path}  ({len(df)} rows, {size_mb:.1f} MB){flag}")

print("\nDone. Copy the 'data' folder into your app's deployment folder/repo.")
print("In your Streamlit app, swap pd.read_csv('data/x.csv') for pd.read_parquet('data/x.parquet').")
