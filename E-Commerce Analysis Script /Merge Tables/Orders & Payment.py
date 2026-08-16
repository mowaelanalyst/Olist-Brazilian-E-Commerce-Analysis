import pandas as pd 

# Load cleaned orders, reviews, and payments datasets
order = pd.read_csv("C_orders.csv")
review = pd.read_csv("C_order_reviws.csv")
payment = pd.read_csv("C_order_payments.csv")


# Aggregate payments to one row per order_id (an order can have multiple payment
# sequences/methods):
# - payment_sequential: take the max to reflect the total number of payment steps
# - payment_type: take the most frequent (mode) method used across the order's payments
# - payment_installments: take the max installment count recorded for the order
# - payment_value: sum across all payment records to get the true total amount paid
payment = payment.groupby("order_id").agg({
    "payment_sequential": "max",
    "payment_type": lambda p : p.mode()[0] if not p.mode().empty else None,
    "payment_installments": "max",
    "payment_value": "sum"
}).reset_index()

# Sort reviews by answer timestamp so that "last" aggregations below reliably
# capture the most recent review activity per order
review = review.sort_values(by= "review_answer_timestamp")

# Aggregate reviews to one row per order_id (an order can be linked to multiple
# review records):
# - review_id: count how many review records exist for the order
# - review_score/title/message: take the last one, based on the sort above
# - review_creation_date: earliest creation date across the order's reviews
# - review_answer_timestamp: latest answer timestamp across the order's reviews
review = review.groupby("order_id").agg({
    "review_id": "count",
    "review_score": "last",
    "review_comment_title": "last",
    "review_comment_message": "last",
    "review_creation_date": "min",
    "review_answer_timestamp": "max"
    
}).reset_index()

# Export the order-level review summary as its own reference table
review.to_csv("Customers_review.csv",index=False)

# Left join orders with the aggregated payment data, keeping every order even if
# it has no matching payment record
orders_paymet = order.merge(payment,on= "order_id",how="left")

# Check how many rows are missing payment data after the merge
null = orders_paymet.isnull().sum()

# Fill missing payment fields for orders with no matching payment record.
# 0 reflects "no payment activity recorded" rather than an imputed/estimated value,
# and "Not Defined" mirrors the existing category already used in the payments data.
orders_paymet["payment_sequential"] = orders_paymet["payment_sequential"].fillna(0)
orders_paymet["payment_type"] = orders_paymet["payment_type"].fillna("Not Defined")
orders_paymet["payment_installments"] = orders_paymet["payment_installments"].fillna(0)
orders_paymet["payment_value"] = orders_paymet["payment_value"].fillna(0)

# Verify the final set of payment_type categories after aggregation and filling
print(orders_paymet["payment_type"].unique())

# Export the merged orders + payments table for the analysis stage
orders_paymet.to_csv("Orders.csv",index=False)
