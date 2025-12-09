import pandas as pd
from datetime import datetime

# Read your original file
df_original = pd.read_csv("sales_data.csv")

# Display original format
print("Original format (first 5 rows):")
print(df_original.head())

# Pivot from wide to long format
# id_vars=['date'] keeps date, all others become item rows
df_long = df_original.melt(
    id_vars=['date'],
    var_name='item_name',
    value_name='sales_count'
)

# Map item names to SKU IDs
item_to_sku = {
    'Milk': 'SKU001',
    'Eggs': 'SKU002',
    'Bread': 'SKU003',
    'Butter': 'SKU004',
    'Cheese': 'SKU005'
}

df_long['item_id'] = df_long['item_name'].map(item_to_sku)

# Convert date format from 2025/12/1 to 2025-12-01
df_long['date'] = pd.to_datetime(df_long['date']).dt.strftime('%Y-%m-%d')

# Reorder columns
df_long = df_long[['date', 'item_id', 'item_name', 'sales_count']]

# Sort by date and item_id for readability
df_long = df_long.sort_values(['date', 'item_id']).reset_index(drop=True)

# Save to sales_data.csv (replace original)
df_long.to_csv("sales_data.csv", index=False)

print(f"\n✅ Conversion Complete!")
print(f"Total rows: {len(df_long)}")
print(f"\nNew format (first 10 rows):")
print(df_long.head(10))
print(f"\nLast 5 rows:")
print(df_long.tail(5))