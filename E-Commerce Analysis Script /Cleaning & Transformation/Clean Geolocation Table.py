import pandas as pd

# Load raw geolocation dataset and previously cleaned customers dataset
geolocation = pd.read_csv("olist_geolocation_dataset.csv")
customers = pd.read_csv("C_customers.csv")

# Ensure 'customer_zip_code_prefix' is cast to string for schema compatibility
customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype(str)

# Missing Values Assessment: Verify missing values across attributes (0 nulls found)
null = geolocation.isnull().sum()

# Deduplication (Exact Matches):
# Inspect full-row duplicate records and drop identical rows
dup = geolocation[geolocation.duplicated(keep=False)].sort_values(by="geolocation_zip_code_prefix").head(50)
geolocation = geolocation.drop_duplicates()

# Data Quality & Cleaning for 'geolocation_zip_code_prefix':
# Convert to string and pad with leading zeros to maintain a standardized 5-digit format
geolocation["geolocation_zip_code_prefix"] = geolocation["geolocation_zip_code_prefix"].astype(str).str.zfill(5)

# Verify zip code data quality: ensure values consist strictly of numeric digits (0 anomalies found)
outlier_value = geolocation[~geolocation["geolocation_zip_code_prefix"].str.contains(r'^\d+$')].shape[0]

# Geographic Coordinate Range Validation:
# Ensure latitude values fall within [-90, 90] and longitude values within [-180, 180] (0 invalid coordinates)
invalid_lat = geolocation[~geolocation["geolocation_lat"].between(-90, 90)].shape[0]
invalid_lng = geolocation[~geolocation["geolocation_lng"].between(-180, 180)].shape[0]

# Data Quality & Cleaning for Categorical Attributes ('geolocation_city' & 'geolocation_state'):
# Strip whitespaces, standardize city names to Title Case, and inspect unique entries
city = geolocation[geolocation["geolocation_city"] != geolocation["geolocation_city"].str.strip()]
geolocation["geolocation_city"] = geolocation["geolocation_city"].str.strip().str.title()
city_uq = geolocation["geolocation_city"].unique()

state = geolocation[geolocation["geolocation_state"] != geolocation["geolocation_state"].str.strip()]
state_uq = geolocation["geolocation_state"].unique()


# DATA QUALITY & GEOLOCATION DEDUPLICATION (PREVENTING DATA EXPLOSION)
#
# ISSUE: Merging 'customers' directly with raw 'geolocation' causes severe row multiplication 
# (~99k rows exploding to ~8.4M rows) due to multiple coordinate records per zip code.
#
# RESOLUTION: Group 'geolocation' by 'geolocation_zip_code_prefix' to ensure a 1:1 relationship.
# Compute the geographic centroid (mean) for Lat/Lng to maintain mapping accuracy, while taking 
# the first record for city/state metadata.


geolocation = geolocation.groupby("geolocation_zip_code_prefix").agg({
    "geolocation_lat": "mean",
    "geolocation_lng": "mean",
    "geolocation_city": "first",
    "geolocation_state": "first"
}).reset_index()

# Export the deduplicated and aggregated geolocation dataset to CSV
geolocation.to_csv("C_geolocation.csv", index=False)
