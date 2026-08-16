import pandas as pd

# Load sellers dataset
sellers = pd.read_csv("olist_sellers_dataset.csv")

# Missing Values Assessment: Verify null counts across all attributes (0 missing values found)
check_null = sellers.isnull().sum()

# Primary Key Uniqueness Check: Ensure 'seller_id' contains no duplicates (0 duplicate rows found)
dup = sellers.duplicated("seller_id").sum()

# Data Quality & Cleaning for 'seller_zip_code_prefix':
# 1. Verify there are no non-positive/invalid numeric zip codes
code = sellers[sellers["seller_zip_code_prefix"] <= 0].shape[0]

# 2. Convert zip codes to string format and pad with leading zeros to ensure a consistent 5-digit length
sellers["seller_zip_code_prefix"] = sellers["seller_zip_code_prefix"].astype(str).str.zfill(5)

# 3. Check for anomalies/outliers: ensure all formatted zip codes consist strictly of numeric digits (0 anomalies found)
outlier_value = sellers[~sellers["seller_zip_code_prefix"].str.contains(r'^\d+$')].shape[0]

# Data Quality & Cleaning for 'seller_city':
# Check for whitespace issues, standardize formatting to Title Case, and inspect unique entries
city = sellers[sellers["seller_city"] != sellers["seller_city"].str.strip()]
sellers["seller_city"] = sellers["seller_city"].str.title()
city_u = sellers["seller_city"].unique()

# Data Quality Check for 'seller_state':
# Verify absence of leading/trailing whitespaces and inspect unique state codes
state = sellers[sellers["seller_state"] != sellers["seller_state"].str.strip()].shape[0]
state_u = sellers["seller_state"].unique()

# Export the cleaned sellers dataset to CSV
sellers.to_csv("C_seller.csv", index=False)
