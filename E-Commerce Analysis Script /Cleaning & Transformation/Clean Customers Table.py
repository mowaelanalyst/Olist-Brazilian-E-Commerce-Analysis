import pandas as pd

# Load the dataset
customers = pd.read_csv("olist_customers_dataset.csv")

# Display general information about the dataframe (data types, non-null counts)
customers.info()

# Check for duplicate customer records based on 'customer_unique_id'
dup = customers[customers.duplicated("customer_unique_id", keep=False)].sort_values(by="customer_unique_id", ascending=False).head(50)

# NOTE ON DUPLICATES:
# Do not drop rows with duplicate 'customer_unique_id' or 'customer_id'.
# In the Olist dataset:
# - 'customer_id': Key for a specific purchase/order transaction.
# - 'customer_unique_id': Unique identifier for the actual person.
# Allowing duplicate 'customer_unique_id' entries is expected and necessary to 
# track repeat purchases across multiple orders.
# -----------------------------------------------------
# Validate zip codes: check for invalid negative values
less_than_zero = customers[customers["customer_zip_code_prefix"] < 0]

# Standardize zip code format: convert to string and pad with leading zeros to ensure a 5-digit format
customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype(str).str.zfill(5)

# Verify zip code data quality: ensure all values consist strictly of numeric digits
outlier_value = customers[customers["customer_zip_code_prefix"].str.contains(r'^\d+$')]

# Check for trailing or leading whitespace in city names
city = customers[customers["customer_city"] != customers["customer_city"].str.strip()].shape[0]

# Standardize city names: convert to Title Case for consistency
customers["customer_city"] = customers["customer_city"].str.title()

# Inspect unique state values to check data quality and detect inconsistencies
state = customers["customer_state"].unique()

# Generate a surrogate primary key ('cut_id') based on 'customer_unique_id' for data modeling
customers["cut_id"] = pd.factorize(customers["customer_unique_id"])[0] + 1

# Export the cleaned dataset to a CSV file
customers.to_csv("C_customers.csv", index=False)
