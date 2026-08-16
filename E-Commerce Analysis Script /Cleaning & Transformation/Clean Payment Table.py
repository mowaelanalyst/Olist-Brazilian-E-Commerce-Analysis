import pandas as pd

# Load order payments dataset
order_payments = pd.read_csv("olist_order_payments_dataset.csv")

# Display dataset summary (data types and non-null check — no missing values found)
order_payments.info()

# Check for composite duplicates across 'order_id' and 'payment_sequential' (0 duplicates found)
dup = order_payments.duplicated(subset=["order_id", "payment_sequential"]).sum()

# Data Quality & Cleaning for 'payment_type':
# 1. Inspect unique values and verify there are no leading/trailing whitespace issues.
payment = order_payments["payment_type"].unique()
end_start_space = order_payments[order_payments["payment_type"] != order_payments["payment_type"].str.strip()]

# 2. Standardize string format: convert to Title Case and replace underscores with spaces (e.g., 'credit_card' -> 'Credit Card')
order_payments["payment_type"] = order_payments["payment_type"].str.title()
order_payments["payment_type"] = order_payments["payment_type"].str.replace("_", " ")

# Data Quality Check for 'payment_installments':
# Identified 2 records with zero installments (<= 0)
less_than_zero_i = order_payments[order_payments["payment_installments"] <= 0]
check_1 = order_payments[order_payments["order_id"].isin(["744bade1fcf9ff3f31d860ace076d422", "1a57108394169c0b47d8f876acc9ba2d"])]

# Impute invalid installments: Replace 0 with 1 for these two credit card transactions. 
# Zero installments is business-invalid, and these orders have no alternative payment logs.
order_payments.loc[order_payments["payment_installments"] == 0, "payment_installments"] = 1

# Data Quality Check for 'payment_value':
# Identify transactions with non-positive values (<= 0)
less_than_zero_v = order_payments[order_payments["payment_value"] <= 0]
check_2 = order_payments[order_payments["order_id"].isin(less_than_zero_v["order_id"].unique())].sort_values(by="order_id")

# Note on zero payment values:
# Zero-value payments were retained as valid entries because they correspond primarily to 'Voucher' 
# or 'Not Defined' payment types, where the overall order cost is covered by other positive payment sequences.

# Verify transaction sequence integrity: ensure max payment sequence per order behaves as expected
print(order_payments.groupby(["order_id", "payment_sequential"]).size().max())

# Export the cleaned/standardized dataset to CSV
order_payments.to_csv("C_order_payments.csv", index=False)
