import pandas as pd
import numpy as np

# Load products dataset
Products = pd.read_csv("olist_products_dataset.csv")

# Missing Values Assessment
null = Products.isnull().sum()

# Handling Missing Values (Imputation Strategy):
# 1. Categorical: Fill missing category names with 'N/A'
Products["product_category_name"] = Products["product_category_name"].fillna("N/A")

# 2. Discrete Numerical Features: Impute missing values with median and cast back to integer
Products["product_name_lenght"] = Products["product_name_lenght"].fillna(Products["product_name_lenght"].median()).astype(int)
Products["product_description_lenght"] = Products["product_description_lenght"].fillna(Products["product_description_lenght"].median()).astype(int)
Products["product_photos_qty"] = Products["product_photos_qty"].fillna(Products["product_photos_qty"].median()).astype(int)

# 3. Continuous Numerical Features: Impute missing physical dimensions/weight with their respective medians
Products["product_weight_g"] = Products["product_weight_g"].fillna(Products["product_weight_g"].median())
Products["product_length_cm"] = Products["product_length_cm"].fillna(Products["product_length_cm"].median())
Products["product_height_cm"] = Products["product_height_cm"].fillna(Products["product_height_cm"].median())
Products["product_width_cm"] = Products["product_width_cm"].fillna(Products["product_width_cm"].median())

# Fix Typo in Column Name: Rename 'lenght' to 'length' for consistency
Products = Products.rename(columns={"product_description_lenght": "product_description_length"})

# Post-imputation verification: confirm no missing values remain after the fillna operations above
null_af = Products.isnull().sum()

# Duplicates Check: Validate primary key uniqueness for 'product_id' (0 duplicates found)
dup = Products.duplicated("product_id").sum()

# Data Quality & Constraint Checks across Numerical and Categorical Attributes:
cat_quality = Products[Products["product_category_name"] != Products["product_category_name"].str.strip()]
len_quality = Products[Products["product_name_lenght"] <= 0]
des_quality = Products[Products["product_description_length"] <= 0]
ph_quality = Products[Products["product_photos_qty"] <= 0]

# Fix Invalid Zero Weights: Replace invalid 0g weights with the column median
wei_quality = Products[Products["product_weight_g"] <= 0]["product_weight_g"]
Products["product_weight_g"] = Products["product_weight_g"].replace(0.0, Products["product_weight_g"].median())

# Check remaining physical dimensions for non-positive values
c_len_quality = Products[Products["product_length_cm"] <= 0]
hei_quality = Products[Products["product_height_cm"] <= 0]
c_wi_quality = Products[Products["product_width_cm"] <= 0]

# Load product category English translation lookup dataset
translation = pd.read_csv("product_category_name_translation.csv")

# Merge Quality Verification:
# Perform outer merge with indicator to evaluate match rates. Unmatched rows align with previously missing category names.
checkmerge = pd.merge(Products, translation, on="product_category_name", how="outer", indicator=True)

# Merge datasets using a left join to attach English translations
Products = Products.merge(translation, on="product_category_name", how="left")

# Post-merge check: flags rows where no English translation was found for the category
# (returns a boolean Series per row, not a total count)
check_null_after_merge = Products["product_category_name_english"].isnull()

# Manual Mapping for Missing Category Translations:
# Map specific untranslated categories ('portateis_cozinha_e_preparadores_de_alimentos' & 'pc_gamer'), then fill remaining with 'N/A'
Products["product_category_name_english"] = np.where(
    (Products["product_category_name_english"].isna()) &
    (Products["product_category_name"] == "portateis_cozinha_e_preparadores_de_alimentos"),
    "Portable Kitchen & Food Processors",
    Products["product_category_name_english"]
)
Products["product_category_name_english"] = np.where(
    (Products["product_category_name_english"].isna()) &
    (Products["product_category_name"] == "pc_gamer"),
    "gaming_pc",
    Products["product_category_name_english"]
)
Products["product_category_name_english"] = Products["product_category_name_english"].fillna("N/A")

# Verify manual imputation accuracy for targeted categories
check_h_done = Products[Products["product_category_name"].isin(["portateis_cozinha_e_preparadores_de_alimentos", "pc_gamer"])][["product_category_name", "product_category_name_english"]]

# Final String Formatting & Cleaning for English Category Names:
# Check for whitespaces, strip underscores, remove trailing duplicate suffixes (e.g., ' 2'), and apply Title Case
en = Products[Products["product_category_name_english"] != Products["product_category_name_english"].str.strip()]

# Inspect the full set of distinct English category names before final formatting,
# to confirm no unexpected duplicate-style variants remain (e.g., 'Category' vs 'Category 2')
uniq_values = Products["product_category_name_english"].unique()

Products["product_category_name_english"] = (
    Products["product_category_name_english"]
    .str.replace("_", " ")
    .str.replace(r" 2$", "", regex=True)
    .str.title()
)

# Export the cleaned products dataset to CSV
Products.to_csv("C_products.csv", index=False)
