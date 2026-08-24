import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged order, item, customer, and review tables
order = pd.read_csv("Orders.csv")
item = pd.read_csv("Order_items.csv")
cust = pd.read_csv("Customers.csv")
reivew = pd.read_csv("Customers_review.csv")

# Bring item-level fields (product/category/price/freight) onto each order row.
# NOTE: this creates one row per item, not per order — order-level columns
# (e.g. payment_value, review_score) will repeat across an order's item rows,
# while item-level columns (price, freight_value) legitimately differ per row.
order = order.merge(item[["order_id","product_id","product_category_name_english","freight_value","price"]],on="order_id",how="left")
order = order.merge(cust[["customer_id","customer_city","customer_unique_id","customer_state"]],on="customer_id",how="left")
order = order.merge(reivew[["order_id","review_score","review_comment_message"]],on="order_id",how="left")

# Exclude canceled/unavailable orders — these never completed a real transaction
order = order[~order["order_status"].isin(["Canceled","Unavailable"])]

order["order_purchase_timestamp"] = pd.to_datetime(order["order_purchase_timestamp"]) 
order["order_delivered_customer_date"] = pd.to_datetime(order["order_delivered_customer_date"]) 

order["Delivery_duration_days"] = (order["order_delivered_customer_date"] - order["order_purchase_timestamp"]).dt.days

order["Year"] = order["order_purchase_timestamp"].dt.year
order["Month"] = order["order_purchase_timestamp"].dt.month

# Restrict to 2016-2017 
order = order[order["Year"] <= 2017].copy()

order = order.sort_values(by=["Year","Month"])
# Overall monthly sales trend (item-level price summed per month) for the line chart
Previous_sales = order.groupby(["Year","Month"])["price"].sum().reset_index()

order = order.sort_values(by=["Year","Month"])
Previous_sales_MoM = order.groupby(["Year","Month"])["price"].sum().reset_index()
# Month-over-month growth as a simple sequential pct_change (compares each month
# to the one immediately before it, regardless of year)
Previous_sales_MoM["Growth_Rate"] = Previous_sales_MoM["price"].pct_change()*100
# Cap out extreme early-period spikes (near-zero prior-month base) as NaN rather
# than plotting a misleading capped number
Previous_sales_MoM["Growth_Rate"] = np.where(
    Previous_sales_MoM["Growth_Rate"] > 1000 ,
    np.nan,
    Previous_sales_MoM["Growth_Rate"] 
)

m_Mom = Previous_sales_MoM["Growth_Rate"].median()
MoM = Previous_sales_MoM[["Year","Month","Growth_Rate"]]

# Baseline period: Jan-Mar 2017, i.e. before the Aprember event
Period = order[(order["Year"] == 2017) & (order["Month"] == 3)].copy()

# Sales per state during the baseline period, to see which states already led
# before Aprember (for context/comparison with the contribution-to-change chart)
Sales_per_state = Period.groupby("customer_state")["price"].sum().reset_index()
Sales_per_state_10 = Sales_per_state.sort_values(by="price",ascending=False).head(10)

# Units sold per category during the baseline period (same purpose as above)
Units_per_category = Period.groupby("product_category_name_english")["order_id"].count().reset_index()
Units_per_category_15 = Units_per_category.sort_values(by="order_id",ascending=False).head(10)

# Full 2017 data, used for the October-vs-Aprember comparison below
Target_Period = order[(order["Year"] == 2017) & (order["Month"] == 4 )].copy() 


# Cumulative comparison: Jan-Mar totals vs Jan-Apr totals, to quantify Aprember's
# overall impact on the year-to-date figures
Total_Sales = Period["price"].sum()
Total_Sales_after_Apr = Target_Period["price"].sum()
Growth_sales_cuse_Apr = (((Total_Sales_after_Apr - Total_Sales)/Total_Sales)*100)

# order_id.nunique() counts distinct orders (item-level fan-out doesn't inflate this)
Total_Orders = Period["order_id"].nunique()
Total_Orders_after_Apr = Target_Period["order_id"].nunique()
Growth_order_cuse_Apr = (((Total_Orders_after_Apr-Total_Orders)/Total_Orders)*100)

# order_id.count() here counts item rows, i.e. total units sold (one row per item)
Total_Sold_units = Period["order_id"].count()
Total_Sold_units_after_Apr = Target_Period["order_id"].count()
Growth_Sold_units_cuse_Apr = (((Total_Sold_units_after_Apr-Total_Sold_units)/Total_Sold_units)*100)

aov = (Total_Sales / Total_Orders )
aov_aetr = (Total_Sales_after_Apr / Total_Orders_after_Apr )

Total_Customers = Period["customer_unique_id"].nunique()
Total_Customers_after_Apr = Target_Period["customer_unique_id"].nunique()
# Growth is measured relative to the pre-event baseline (Total_Customers),
# consistent with how sales/orders growth is calculated above
Growth_Cust_cuse_Apr = (((Total_Customers_after_Apr-Total_Customers)/Total_Customers)*100)

New_customers = order.groupby("customer_unique_id")["order_purchase_timestamp"].min().reset_index()
New_customers_Apr = New_customers[(New_customers["order_purchase_timestamp"].dt.year == 2017) &
(New_customers["order_purchase_timestamp"].dt.month == 4)]
Total_New_Customer = New_customers_Apr["customer_unique_id"].shape[0]
pre_of_new = f"{((Total_New_Customer/Total_Customers_after_Apr)*100):,.2f}"


avg_delivery_days = Period["Delivery_duration_days"].mean()
avg_delivery_days_after_Apr = Target_Period["Delivery_duration_days"].mean()

frequency = Total_Sold_units / Total_Customers
frequency_after_Apr = Total_Sold_units_after_Apr / Total_Customers_after_Apr

# freight_value is item-level (varies per product within the same order), so it
# is summed directly across all item rows without deduplicating on order_id —
# deduplicating here would drop real freight amounts for multi-item orders
Total_freight_cost = Period["freight_value"].sum()
Avg_freight_cost = Period["freight_value"].mean()
Total_freight_cost_after_Apr = Target_Period["freight_value"].sum()
Avg_freight_cost_after_Apr = Target_Period["freight_value"].mean()
Growth_cost_cuse_Apr = (((Total_freight_cost_after_Apr-Total_freight_cost)/Total_freight_cost)*100)

# price is also item-level, so its median is computed directly on the item rows
# (no dedup) to reflect the true distribution of individual item prices
avg_price = Period["price"].median()
avg_price_after_Apr = Target_Period["price"].median()

# review_score is order-level (one score per order, repeated across that
# order's item rows), so it requires dropping duplicate order_id rows first
# to avoid counting the same review multiple times

No_dup_period = Period.drop_duplicates("order_id").copy()
No_dup_Target_period = Target_Period.drop_duplicates("order_id").copy()

avg_reivew_score = No_dup_period[No_dup_period["Month"] == 3 ]["review_score"].mean()
avg_reivew_score_after_Apr = No_dup_Target_period[No_dup_Target_period["Month"] == 4 ]["review_score"].mean()


# --- Dashboard 2: KPI summary text panel (Jan-Mar vs Jan-Apr) ---
f , k = plt.subplots(1,1,figsize=(15,10))
plt.axis("off")


f.text(0.5, 0.94, " 2017 Performance  (Mar vs Apr)", fontsize=15, fontweight="bold", ha="center")


col1, col2, col3, col4 = 0.125, 0.375, 0.625, 0.875
y_head = 0.84
y_row1 = 0.68
y_row2 = 0.48
y_row3 = 0.28


f.text(col1, y_head, "[ Sales & Revenue ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col1, y_row1, f"Total Sales:\nMar: {Total_Sales:,.0f}  |  Apr: {Total_Sales_after_Apr:,.0f}\n({Growth_sales_cuse_Apr:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f.text(col1, y_row2, f"AOV:\nMar: {aov:,.1f}  |  Apr: {aov_aetr:,.1f}", fontsize=10, fontweight="bold", ha="center")
f.text(col1, y_row3, f"Median Item Price:\nMar: {avg_price:,.1f}  |  Apr: {avg_price_after_Apr:,.1f}", fontsize=10, fontweight="bold", ha="center")


f.text(col2, y_head, "[ Orders & Volume ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col2, y_row1, f"Total Orders:\nMar: {Total_Orders:,.0f}  |  Apr: {Total_Orders_after_Apr:,.0f}\n({Growth_order_cuse_Apr:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f.text(col2, y_row2, f"Total Sold Units:\nMar: {Total_Sold_units:,.0f}  |  Apr: {Total_Sold_units_after_Apr:,.0f}\n({Growth_Sold_units_cuse_Apr:+.1f}%)", fontsize=10, fontweight="bold", ha="center")

f.text(col3, y_head, "[ Customers & Review ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col3, y_row1, f"Total Customers:\nMar: {Total_Customers:,.0f}  |  Apr: {Total_Customers_after_Apr:,.0f}\n({Growth_Cust_cuse_Apr:+.1f}%)\n New Customers: {Total_New_Customer}:({pre_of_new})%", fontsize=10, fontweight="bold", ha="center")
f.text(col3, y_row2, f"Purchase Frequency:\nMar: {frequency:.2f}  |  Apr: {frequency_after_Apr:.2f}", fontsize=10, fontweight="bold", ha="center")
f.text(col3, y_row3, f"Avg Review Score:\nMar: {avg_reivew_score:.2f}  |  Apr: {avg_reivew_score_after_Apr:.2f}", fontsize=10, fontweight="bold", ha="center")


f.text(col4, y_head, "[ Logistics & Freight ]", fontsize=11, fontweight="bold", ha="center", color="navy")
f.text(col4, y_row1, f"Total Freight Cost:\nMar: {Total_freight_cost:,.0f}  |  Apr: {Total_freight_cost_after_Apr:,.0f}\n({Growth_cost_cuse_Apr:+.1f}%)", fontsize=10, fontweight="bold", ha="center")
f.text(col4, y_row2, f"Avg Freight Cost:\nMar: {Avg_freight_cost:,.1f}  |  Apr: {Avg_freight_cost_after_Apr:,.1f}", fontsize=10, fontweight="bold", ha="center")
f.text(col4, y_row3, f"Avg Delivery Days:\nMar: {avg_delivery_days:.1f}d  |  Apr: {avg_delivery_days_after_Apr:.1f}d", fontsize=10, fontweight="bold", ha="center")

# --- Dashboard 3: Overall MoM growth rate and monthly sales trend (2016-2017) ---
fig , axse = plt.subplots(1,2,figsize=(15,10))


sns.lineplot(data=MoM,
x="Month",
y="Growth_Rate",
hue="Year",
ax=axse[0],
marker="o"
)
axse[0].set_title(f"Growth Rate MoM (Median:{m_Mom:,.2f}% )")

for x, y in zip(MoM["Month"], MoM["Growth_Rate"]):
    axse[0].annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,8), ha='center')
    
sns.lineplot(data=Previous_sales,
x="Month",
y="price",
hue="Year",
ax=axse[1],
marker="o"
)

axse[1].set_title("Total Sales Overtime")

for x, y in zip(Previous_sales["Month"], Previous_sales["price"]):
        axse[1].annotate(f'{y:,.0f}$', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center')

plt.tight_layout(h_pad=3.0)

plt.show()
