# Olist-Brazilian-E-Commerce-Analysis
Clean And Analyzing Olist Brazilian E-Commerce Dataset And Create Insights  
## Data Cleaning & Merging

This project uses the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 8 relational tables covering ~99K orders (2016–2018): orders, order items, customers, products, sellers, payments, reviews, and geolocation.

Below are the key data quality issues identified and how each was handled, table by table.

### `orders`
- Checked all six logical timestamp relationships (purchase → approval → carrier dispatch → customer delivery) for inconsistencies. Found violations ranging from 32 to 1,356 rows across different comparisons.
- **Decision:** Rows were not dropped. Instead, invalid rows are flagged and excluded specifically when computing delivery-duration or time-based KPIs, preserving order volume and non-temporal data for other analyses.
- Missing values in delivery-related date columns were retained as `NaT` where they correspond to non-delivered order statuses (e.g., canceled), rather than imputed.

### `customers`
- Identified that `customer_id` is transaction-level (unique per order) while `customer_unique_id` identifies the actual person. Rows that looked like duplicates on `customer_unique_id` were legitimate — deduplicating on it would have broken referential integrity with the `orders` table.
- Standardized zip code prefixes to a consistent 5-digit string format, explicitly preserving leading zeros (a `dtype=str` fix was required on every later CSV re-read to prevent pandas from re-inferring the column as an integer).

### `order_items`
- Distinguished between `price <= 0` (a true data error) and `freight_value == 0` (a legitimate free-shipping case) — applying `< 0` rather than `<= 0` for freight validation.

### `products`
- Missing numeric attributes (weight, dimensions, description length) imputed with column medians; missing category imputed as `"N/A"`.
- Cross-referenced against the category translation table; two untranslated categories were mapped manually after inspection.
- Fixed a regex range bug (`A-z` instead of `A-Z`) that was silently allowing stray symbols through a multilingual text-cleaning step.

### `order_payments`
- Aggregated to one row per order (orders can have multiple payment steps/methods), using `sum` for total payment value, `max` for installment/sequence counts, and mode for the dominant payment method.
- Investigated and resolved 2 rows with `payment_installments == 0` (business-invalid) based on manual order-level inspection.

### `order_reviews`
- Discovered `review_id` is not a unique key (a review can link to multiple orders) — aggregation logic was built around `order_id` instead.
- Cleaned free-text fields with a Unicode-aware regex (Latin, Latin-extended, Arabic ranges) to strip noise while preserving valid multilingual content.

### `geolocation`
- Multiple lat/lng records exist per zip code prefix. A direct merge with `customers` would have exploded ~99K rows into ~8.4M. Resolved by aggregating to one row per zip prefix (mean centroid for coordinates, first value for city/state) before merging.

### Merging Stage
- Built customer-level, order-level, and item-level tables tailored to each analysis need, rather than one fully flattened table — deliberately avoiding a fan-out bug where merging order-level financial data (`payment_value`) with item-level rows would have inflated every revenue metric by the item count per order.
- For missing coordinates after merging, applied `ffill`/`bfill` within `groupby` (by customer or by seller city) to propagate real, verified coordinates instead of introducing synthetic values via mean/median imputation on spatial data.

**Tools:** Python, pandas, NumPy
