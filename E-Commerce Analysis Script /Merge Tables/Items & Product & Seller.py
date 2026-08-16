import pandas as pd 
 
# Load cleaned order items, products, sellers, and geolocation datasets.
# dtype=str is specified on the zip code columns to preserve leading zeros.
item = pd.read_csv("C_order_items.csv")
pro = pd.read_csv("C_products.csv")
seller = pd.read_csv("C_seller.csv",dtype={"seller_zip_code_prefix": str})
geolo = pd.read_csv("C_geolocation.csv",dtype={"geolocation_zip_code_prefix": str})


# Left join sellers with geolocation on the zip code prefix, keeping every seller
# even when no matching geolocation entry exists
seller_geolo = seller.merge(geolo,
left_on="seller_zip_code_prefix",
right_on="geolocation_zip_code_prefix",
how="left"
)
# Columns to retain from the seller+geolocation merge for downstream use
seller_needed_columns = ["seller_id","seller_city","seller_state","geolocation_lat","geolocation_lng"]

# Fill missing coordinates for a seller by borrowing coordinates from another
# seller in the same city (grouped by seller_city, since each seller_id is
# already a single row here and cannot fill from itself)
seller_geolo["geolocation_lng"] = seller_geolo.groupby("seller_city")["geolocation_lng"].ffill().bfill()
seller_geolo["geolocation_lat"] = seller_geolo.groupby("seller_city")["geolocation_lat"].ffill().bfill()

# Keep only the needed seller+location columns
seller = seller_geolo[seller_needed_columns]

# Left join order items with the seller (+location) data on seller_id,
# keeping every item row even if seller info is missing
item_seller = item.merge(seller,
left_on="seller_id",
right_on="seller_id",
how="left"
)

# Left join the result with product data on product_id, keeping every item row
# even if product info is missing
item_pro = item_seller.merge(pro,
on="product_id",
how="left",

)


# Columns kept for the final item-level table used in the analysis stage
item_needed_columns = ['order_id', 'order_item_id', 'product_id', 'seller_id',
       'shipping_limit_date', 'price', 'freight_value', 'seller_city',
       'seller_state', 'geolocation_lat', 'geolocation_lng',
       'product_category_name_english','product_name_lenght',
       'product_description_length', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm']
item_pro = item_pro[item_needed_columns]

# Export the final item-level table (order items + seller + product info)
item_pro.to_csv("Order_items.csv",index=False)
