import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged order, item
order = pd.read_csv("Orders.csv")
item = pd.read_csv("Order_items.csv")


# Bring item-level fields (price) onto each order row.
# NOTE: this creates one row per item, not per order 

order = order.merge(item[["order_id","price"]],on="order_id",how="left")


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
Sales_overtime = order.groupby(["Year","Month"])["price"].sum().reset_index()

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
    
sns.lineplot(data=Sales_overtime,
x="Month",
y="price",
hue="Year",
ax=axse[1],
marker="o"
)

axse[1].set_title("Total Sales Overtime")

for x, y in zip(Sales_overtime["Month"], Sales_overtime["price"]):
        axse[1].annotate(f'{y:,.0f}$', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center')

plt.tight_layout(h_pad=3.0)

plt.show()
