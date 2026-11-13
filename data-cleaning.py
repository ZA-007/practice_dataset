
''''' for classification
1. Load dataset
2. Inspect data (head, info, nulls)
3. Drop irrelevant columns (e.g., customerID)
4. Handle missing values
5. Fix data types
6. Encode categorical columns and y if needed 
7. Separate X and y
9. Train-test split
10. Build models
11. Evaluate models'''''


# Check missing values
print(data.isnull().sum())

#if a small numbers of rows contain missing values considr removing them
# Remove rows containing missing values
data = data.dropna()

#drop duplicates
data = data.drop_duplicates()


# Replace missing numerical values with the median
data["column_name"] = data["column_name"].fillna(
    data["column_name"].median()
)

# Replace missing categorical values with the most common value
data["column_name"] = data["column_name"].fillna(
    data["column_name"].mode()[0]
)

#Median: useful for numerical columns, especially when values are skewed.
#Mode: useful for categorical columns.

#check for outliers
# Calculate Q1 and Q3
Q1 = data["column_name"].quantile(0.25)
Q3 = data["column_name"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower and upper boundaries
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[
    (data["column_name"] < lower) |
    (data["column_name"] > upper)
]

print(outliers)
#u can use box plot to see

#u can aslo check for outliers with a box plot :
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=data["column_name"])

plt.title("Monthly Charges Boxplot")
plt.show()

#or since outliers are only for numerical values and if u have mnay features that are numerical do :
numeric_columns = data.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# Remove outliers
data = data[
    (data["column_name"] >= lower) &
    (data["column_name"] <= upper)
]

# Cap outliers
data["column_name"] = data["column_name"].clip(
    lower=lower,
    upper=upper
)

# Convert a column to numerical
data["column_name"] = pd.to_numeric(data["column_name"])

# Convert a column to dates
data["date"] = pd.to_datetime(data["date"])

# Convert a column to category
data["column_name"] = data["column_name"].astype("category")

#scaling numerical features
#Models such as KNN, SVM and K-Means, logistic regression can be affected by these differences.
#using standardscaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#feature selection
#When you have unnecessary features, irrelevant columns or columns that could cause problems.
#For example, an ID column that simply identifies each customer usually isn't useful for predicting their spending.

X = data.drop(
    ["target", "CustomerID"],
    axis=1
)
#Here you need to replace: "target" with the name of your target column. "CustomerID" with the name of an irrelevant column you want to remove

#You can use LabelEncoder to turn target labels into numbers:
from sklearn.preprocessing import LabelEncoder

"""encoder = LabelEncoder()
y = encoder.fit_transform(y)
Select the features you want.
One-hot encode categorical columns such as Gender.
Scale the features.""""

# Select features
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
X = df[features].copy()

# One-hot encode Gender
X = pd.get_dummies(X, columns=['Gender'], dtype=int)

#or label encoding can also be done for classification instead of hot encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

#or hot encoding is fine:
data = pd.get_dummies(data, columns=['gender'], dtype=int)

'''''For your exam, use this rule:

Algorithm	Categorical Features
Decision Tree	Label Encoding ✅
Random Forest	Label Encoding ✅
KNN	Label Encoding ✅ (or One-Hot)
Linear Regression	One-Hot Encoding ✅
Logistic Regression	One-Hot Encoding ✅'''''

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

''''Distribution	Interpretation
50/50	Perfectly balanced
60/40	Generally balanced
70/30	Mild to moderate imbalance
80/20	Imbalanced
90/10	Strongly imbalanced
95/5	Severely imbalanced''''
#to see if churn is imbalanced or not
print(data["Churn"].value_counts())
print(data["Churn"].value_counts(normalize=True) * 100)