import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#Load and explore the data
data = pd.read_csv('Groceries_dataset.csv')
print(data.head())
print(data.info())
print(data.columns)
print(data['Member_number'].head(20)) #check theres no "C" for cancellation like invoice number had


# Number of unique transactions
num_transactions = data[ ['Member_number', 'Date'] ].drop_duplicates().shape[0] #we combined date with member number here because the same member can shop on multiple dates

print( "Number of transactions:", num_transactions)

# Number of unique products
num_products = data['itemDescription'].nunique()

print("Number of products:",num_products)

#cleaning the data
data['itemDescription'] = data['itemDescription'].str.strip()

data.dropna(axis = 0, subset =['Member_number'], inplace = True)

data['Member_number'] = data['Member_number'].astype('str')

#this shows us that a customer can appear multiple times on different shopping dates
print(data.groupby('Member_number')['Date'].nunique().sort_values(ascending=False).head())

#we create our own trasnaction id by combining the date with member number
data["TransactionID"] = (
    data["Member_number"].astype(str)
    + "_"
    + data["Date"].astype(str)
)

print(
    data[
        ["Member_number",
         "Date",
         "TransactionID"]
    ].head()
)
# The Groceries dataset does not contain a dedicated
# transaction identifier such as InvoiceNo.

# Therefore, a TransactionID was created by combining
# Member_number and Date. This creates a unique
# identifier for each shopping basket and allows the
# data to be transformed into a market basket matrix
# suitable for Apriori analysis.

#now we actually build the market basket
basket = (
    data.groupby(
        ['TransactionID', 'itemDescription']
    )['itemDescription']
    .count()
    .unstack()
    .fillna(0)
)

print(basket.shape)



#hot encoding
def hot_encode(x):
    if (x <= 0):
        return 0
    if (x >= 1):
        return 1

basket_encoded = basket.map(hot_encode)
basket = basket_encoded

# Building the model
frq_items = apriori(basket, min_support=0.005, use_colnames=True)


# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)


# Sort and display strongest rules
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(10))

#too see the full dataframe better
pd.set_option('display.max_columns', None)

print(
    rules[
        ['antecedents',
         'consequents',
         'support',
         'confidence',
         'lift']
    ]
)

#FOR THE min_support= 0.005, when the support was higher like 0.05 we got an empty dataframe, this tells us that no item combination occured in at least 5% of all transactions
#which indicates customer purchases were highly diverse and product combinations occured infrequently
#therefore, the threshold was reduced to 0.5% to identify meanignful purchasing relationships

#how to intrpret the results
# ==========================================
# APRIORI RESULTS INTERPRETATION
# ==========================================

# antecedents:
# The starting product(s) in the rule.

# Example:
# antecedent = {sausage}

# Means:
# The customer purchased sausage.


# consequents:
# The product(s) associated with the antecedent.

# Example:
# consequent = {yogurt}

# Means:
# The rule is:

# sausage -> yogurt

# In other words, customers who buy sausage
# may also buy yogurt.


# ==========================================
# SUPPORT
# ==========================================

# Support measures how frequently the rule
# appears within all transactions.

# Formula:
# Support = Transactions containing both items
#           divided by Total Transactions

# Example:
# Support = 0.0057

# Means:
# 0.57% of all shopping baskets contained
# both products.

# Higher support means the rule occurs more
# frequently.


# ==========================================
# CONFIDENCE
# ==========================================

# Confidence measures how likely customers
# are to buy the consequent after buying the
# antecedent.

# Formula:
# Confidence =
# Support(A and B) / Support(A)

# Example:
# sausage -> yogurt

# Confidence = 0.095

# Means:
# 9.5% of customers who purchased sausage
# also purchased yogurt.

# Higher confidence means a stronger
# purchasing relationship.


# ==========================================
# LIFT
# ==========================================

# Lift is usually the most important metric.

# It measures whether the association occurs
# more often than expected by chance.

# Formula:
# Lift =
# Confidence(A -> B) / Support(B)

# Interpretation:

# Lift > 1
# Positive association

# Lift = 1
# No relationship

# Lift < 1
# Negative relationship

# Example:
# Lift = 1.11

# Means:
# Customers purchasing sausage are about
# 11% more likely to purchase yogurt than
# would be expected by random chance.

# Higher lift indicates stronger association.


# ==========================================
# HOW TO INTERPRET RULES
# ==========================================

# Example Rule:

# antecedent:
# {sausage}

# consequent:
# {yogurt}

# support:
# 0.0057

# confidence:
# 0.095

# lift:
# 1.11

# Interpretation:

# Approximately 0.57% of all shopping baskets
# contained both sausage and yogurt.
# Among customers purchasing sausage,
# approximately 9.5% also purchased yogurt.
# The lift value of 1.11 indicates a weak
# positive association between these products.
# Customers purchasing sausage are slightly
# more likely to purchase yogurt than expected
# by chance.