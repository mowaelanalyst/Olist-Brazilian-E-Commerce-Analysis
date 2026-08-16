import pandas as pd

# Load order items dataset
order_items = pd.read_csv("olist_order_items_dataset.csv")


# Verify primary key uniqueness: check for composite duplicates across 'order_id' and 'order_item_id'
dup = order_items.duplicated(subset=["order_id", "order_item_id"]).sum()

# Note: 'shipping_limit_date' requires type conversion to datetime, 
# which will be deferred to the analysis phase.

# Validate financial metrics: ensure prices are positive and freight values are non-negative
less_than_zero = order_items[order_items["price"] <= 0]
less_than_zero_c = order_items[order_items["freight_value"] < 0].shape[0]

# Export the cleaned/validated dataset
order_items.to_csv("C_order_items.csv", index=False)
