import pandas as pd

# Load order reviews dataset
order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")

# Display full column text without truncation for inspecting review messages
pd.set_option("display.max_colwidth", None)

# Missing Values Assessment & Handling:
# Check total null count per column, then impute missing titles and messages with 'Unknown'
check_null = order_reviews.isnull().sum()
order_reviews["review_comment_title"] = order_reviews["review_comment_title"].fillna("Unknown")
order_reviews["review_comment_message"] = order_reviews["review_comment_message"].fillna("Unknown")

# Duplicates Check:
# Note: 'review_id' is not strictly unique as a single review can be linked to multiple orders.
dup = order_reviews[order_reviews.duplicated("review_id", keep=False)].sort_values(by="review_id", ascending=False).head(50)

# Data Quality Check: Validate that all review scores fall within the valid range [1, 5]
rev = order_reviews[~order_reviews["review_score"].between(1, 5)].shape[0]

# Clean & Standardize 'review_comment_title':
# 1. Strip special characters/emojis (retaining English, Portuguese/Latin accents, Arabic, and spaces)
# 2. Trim whitespaces and replace empty strings resulting from cleaning with 'Unknown'
order_reviews["review_comment_title"] = (order_reviews["review_comment_title"]
                                         .str.replace(r"[^a-zA-Z\u00C0-\u024F\u0600-\u06FF\s]", "", regex=True)
                                         .str.strip())
order_reviews["review_comment_title"] = order_reviews["review_comment_title"].replace("", "Unknown")

# Verify title data quality post-cleaning
tit = (order_reviews[~order_reviews["review_comment_title"]
       .str.contains(r"^[a-zA-Z\u00C0-\u024F\u0600-\u06FF\s]+$", na=False)]["review_comment_title"].head(50))

# Clean & Standardize 'review_comment_message':
# Apply same Regex cleaning rules to remove noise/special characters while preserving valid multilingual text
order_reviews["review_comment_message"] = (order_reviews["review_comment_message"]
                                           .str.replace(r"[^a-zA-Z\u00C0-\u024F\u0600-\u06FF\s]", "", regex=True)
                                           .str.strip())
order_reviews["review_comment_message"] = order_reviews["review_comment_message"].replace("", "Unknown")

# Verify message data quality post-cleaning
mass = (order_reviews[~order_reviews["review_comment_message"]
        .str.contains(r"^[a-zA-Z\u00C0-\u024F\u0600-\u06FF\s]+$", na=False)]["review_comment_message"].shape[0])

# Datetime Conversion & Sequence Validation:
# Convert review dates to datetime format and ensure creation date never succeeds the answer timestamp (0 logical errors)
order_reviews["review_creation_date"] = pd.to_datetime(order_reviews["review_creation_date"])
order_reviews["review_answer_timestamp"] = pd.to_datetime(order_reviews["review_answer_timestamp"])
date = order_reviews[order_reviews["review_creation_date"] > order_reviews["review_answer_timestamp"]].shape[0]

# Export the cleaned reviews dataset to CSV
order_reviews.to_csv("C_order_reviws.csv", index=False)
