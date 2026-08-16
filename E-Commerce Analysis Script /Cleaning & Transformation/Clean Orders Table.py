import pandas as pd

# Load orders dataset
orders = pd.read_csv("olist_orders_dataset.csv")

# Display all columns without truncation for thorough data inspection
pd.set_option("display.max_columns", None)

# Missing Values Assessment:
# NOTE: Retaining NULLs in date columns ('order_delivered_carrier_date', 'order_approved_at', 'order_delivered_customer_date').
# Missing timestamps represent non-occurring events (e.g., canceled/undelivered orders), 
# and imputing them would distort time-series and fulfillment latency analysis.
null = orders.isnull().sum()

# Primary Key Uniqueness Verification: Check for duplicate order IDs (0 duplicates found)
dup = orders.duplicated("order_id").sum()

# Data Quality & Cleaning for 'order_status':
# Inspect unique statuses, standardize to Title Case, and verify there are no leading/trailing whitespace issues

# Initial inspection of all distinct status values before any cleaning,
# to catch unexpected categories or spelling inconsistencies early
Q_order_status = orders["order_status"].unique()
orders["order_status"] = orders["order_status"].str.title()

# Whitespace check performed AFTER standardizing to Title Case,
# to confirm the formatted values are also free of leading/trailing spaces
s_e_space = orders[orders["order_status"] != orders["order_status"].str.strip()]

# Chronological Sequence & Data Integrity Checks across Timestamp Attributes:
# Validate logical progression: Purchase -> Approval -> Carrier Delivery -> Customer Delivery

# 1. Purchase Date vs. Approval Date
pu_not_biger_app = orders[orders["order_purchase_timestamp"] > orders["order_approved_at"]].shape[0]

# 2. Purchase Date vs. Carrier Delivery Date
pu_not_biger_carr = orders[orders["order_purchase_timestamp"] > orders["order_delivered_carrier_date"]].shape[0]

# 3. Purchase Date vs. Customer Delivery Date
pu_not_biger_deliv = orders[orders["order_purchase_timestamp"] > orders["order_delivered_customer_date"]].shape[0]

# 4. Approval Date vs. Carrier Delivery Date
app_not_biger_carr = orders[orders["order_approved_at"] > orders["order_delivered_carrier_date"]].shape[0]

# 5. Approval Date vs. Customer Delivery Date
app_not_biger_deliv = orders[orders["order_approved_at"] > orders["order_delivered_customer_date"]].shape[0]

# 6. Carrier Delivery Date vs. Customer Delivery Date
carr_not_biger_deliv = orders[orders["order_delivered_carrier_date"] > orders["order_delivered_customer_date"]].shape[0]


# DATA INTEGRITY WARNING & ANALYTICAL DECISION:
# Multiple chronological violations were detected across timestamp columns.
# DECISION: Retain these records to preserve overall order volume and transactional data.
# However, flag these specific timestamp fields as partially unreliable during EDA/Time-Series analysis, 
# and exclude/adjust them when calculating fulfillment latency or SLA metrics.


# Export the cleaned orders dataset to CSV
orders.to_csv("C_orders.csv", index=False)
