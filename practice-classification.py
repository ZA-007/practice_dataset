import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier, AdaBoostClassifier)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv("telco_data.csv")


#explore data
print(data.head())
print(data.shape) #rows and columns
print(data.info())# column names and data types
print(data.describe())  #stats info
print('missing values',data.isnull().sum()) #missing values
print('duplicates',data.duplicated().sum())

#dropping irrelevant columns
data.drop("customerID", axis=1, inplace=True)
data.drop("tenure", axis=1, inplace=True) #dropped it because it has a high percentage of missing values which could increase the risk of introducing bias through extensive imputation


print("shows first few values and the data types for every column")
for col in data.columns:
    print("\n", col)
    print(data[col].head())
    print(data[col].dtype)


#convert total charges from stirng to numeric data type
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

missing_percentage = (
data.isnull().sum() / len(data)
) * 100

print('missing percentage',missing_percentage)

print(list(data.columns))

#handling missing values
data['gender'] = data['gender'].fillna(
    data['gender'].mode()[0]
)

data["Partner"] = data["Partner"].fillna(
    data["Partner"].mode()[0]
)

data["InternetService"] = data["InternetService"].fillna(
    data["InternetService"].mode()[0]
)

data["StreamingTV"] = data["StreamingTV"].fillna(
    data["StreamingTV"].mode()[0]
)

data["MonthlyCharges"] = data["MonthlyCharges"].fillna(
    data["MonthlyCharges"].median()
)

data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median())

print("churn",data["Churn"].unique()) #this shows yes and no meaning we have to encode it to 0 and 1

#encode columns that need it and target variable
encoder = LabelEncoder()
# Encode categorical columns

encoder = LabelEncoder()

data["gender"] = encoder.fit_transform(data["gender"])

data["Partner"] = encoder.fit_transform(data["Partner"])

data["Dependents"] = encoder.fit_transform(data["Dependents"])

data["PhoneService"] = encoder.fit_transform(data["PhoneService"])

data["MultipleLines"] = encoder.fit_transform(data["MultipleLines"])

data["InternetService"] = encoder.fit_transform(data["InternetService"])

data["OnlineSecurity"] = encoder.fit_transform(data["OnlineSecurity"])

data["OnlineBackup"] = encoder.fit_transform(data["OnlineBackup"])

data["DeviceProtection"] = encoder.fit_transform(data["DeviceProtection"])

data["TechSupport"] = encoder.fit_transform(data["TechSupport"])

data["StreamingTV"] = encoder.fit_transform(data["StreamingTV"])

data["StreamingMovies"] = encoder.fit_transform(data["StreamingMovies"])

data["Contract"] = encoder.fit_transform(data["Contract"])

data["PaperlessBilling"] = encoder.fit_transform(data["PaperlessBilling"])

data["PaymentMethod"] = encoder.fit_transform(data["PaymentMethod"])

data["Churn"] = encoder.fit_transform(data["Churn"])


#to see if churn is imbalanced or not
print(data["Churn"].value_counts())
print(data["Churn"].value_counts(normalize=True) * 100)

print("churn data type",data["Churn"].dtype)

#checking for outliers using box plot
sns.boxplot(x=data["MonthlyCharges"])
plt.title("Monthly Charges Boxplot")
plt.show()

sns.boxplot(x=data["TotalCharges"])
plt.title("Total Charges Boxplot")
plt.show()

#checking for outliers numerically
Q1 = data["MonthlyCharges"].quantile(0.25)
Q3 = data["MonthlyCharges"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[(data["MonthlyCharges"] < lower) | (data["MonthlyCharges"] > upper)]

print(len(outliers))




#then seperate x input features and y the target
x = data.drop("Churn", axis=1) #axis 1 drops a column rather than row
y = data["Churn"]

print('data types after encoding and converting total charges to numeric', data.dtypes)


X_train, X_test, y_train, y_test = train_test_split(x , y ,
                                                    test_size= 0.2,
                                                    random_state=1,
                                                    stratify=y)

print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])


#first traditional model: decision tree
decision_tree= DecisionTreeClassifier( random_state=1, class_weight='balanced')
decision_tree.fit(X_train, y_train)
decision_predictions = decision_tree.predict(X_test)

decision_accuracy = accuracy_score(
    y_test,
    decision_predictions
)

decision_f1 = f1_score(
    y_test,
    decision_predictions)

print(f"Decision Tree Accuracy: {decision_accuracy:.2f}")
print(f"Decision Tree F1 Score: {decision_f1:.2f}")


#advanced model, random forest bagging
bagging_model = RandomForestClassifier(
    n_estimators=200,
    random_state=1,
    n_jobs=-1,
    class_weight='balanced'
)

bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(
    y_test,
    bagging_predictions
)
bagging_f1 = f1_score(
    y_test,
    bagging_predictions
)
print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")

''''Random Forest is a bagging ensemble technique that combines multiple decision trees trained on different subsets of data. The final prediction is obtained using majority voting, reducing overfitting and improving generalisation'''

''''Random forest performed better so far because Random Forest combines multiple decision trees using bagging, which reduces overfitting and improves generalisation performance.
More accurate churn predictions allow the company to identify at-risk customers more effectively and improve retention strategies.'''''


# Calculate how many predictions each model got correct
decision_correct = (y_test == decision_predictions).sum()
random_forest_correct = (y_test == bagging_predictions).sum()

print("Decision Tree correct predictions:", decision_correct)
print("Random Forest correct predictions:", random_forest_correct)


#deicison tree confusion matrix
cm_dt = confusion_matrix(y_test, decision_predictions)

display_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["No", "Yes"]
)

display_dt.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

#random forest confusion matrix

cm_rf = confusion_matrix(y_test, bagging_predictions)

display_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["No", "Yes"]
)

display_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

# Random Forest feature importance from most to least important.

feature_importance = pd.Series(
    bagging_model.feature_importances_,
    index=x.columns
)

feature_importance = feature_importance.sort_values(ascending=False)

print(feature_importance)


# Decision Tree feature importance

feature_importance_dt = pd.Series(
    decision_tree.feature_importances_,
    index=x.columns
)

feature_importance_dt = feature_importance_dt.sort_values(ascending=False)

print(feature_importance_dt)


#feature importance graph
# Plot Random Forest feature importance

plt.figure(figsize=(10, 6))

feature_importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Hardware Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

#cross validation, Cross-validation helps us check whether the models perform consistently across different splits.

# Set up repeated stratified cross-validation
cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=1
)

# Test Decision Tree using cross-validation
dt_cv_accuracy = cross_val_score(
    decision_tree,
    X_train,
    y_train,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

print("Decision Tree CV Scores:", dt_cv_accuracy)
print("Decision Tree Mean Accuracy:", dt_cv_accuracy.mean())
print("Decision Tree Standard Deviation:", dt_cv_accuracy.std())

# Test Random Forest using cross-validation
rf_cv_accuracy = cross_val_score(
    bagging_model,
    X_train,
    y_train,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

print("Random Forest CV Scores:", rf_cv_accuracy)
print("Random Forest Mean Accuracy:", rf_cv_accuracy.mean())
print("Random Forest Standard Deviation:", rf_cv_accuracy.std()) #Standard deviation tells us how much the accuracy scores varied across the 30 rounds.
#A lower standard deviation means more consistent results across the different splits.
#Therefore, Random Forest was not only more accurate, but also slightly more consistent.

#compare the results
print("Decision Tree CV Accuracy:", dt_cv_accuracy.mean())
print("Random Forest CV Accuracy:", rf_cv_accuracy.mean())

'''''Cross-validation was used to assess how well the model generalises to unseen data. Instead of evaluating the model on a single train-test split, 
the dataset was divided into multiple folds and the model was trained and tested repeatedly. The average accuracy across all folds provides a more reliable estimate of model performance and helps identify overfitting.'''

'''''TotalCharges, Contract type, and MonthlyCharges were identified as the most influential variables affecting customer churn. 
This suggests that customer expenditure and contract commitments play a significant role in predicting whether a customer is likely to leave the company.'''''



#i could do adaboost instead of random forest :
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=1
    ),
    n_estimators=200,
    random_state=1
)

boosting_model.fit(X_train, y_train)

boost_predictions = boosting_model.predict(X_test)

boost_accuracy = accuracy_score(
    y_test,
    boost_predictions
)

boost_f1 = f1_score(
    y_test,
    boost_predictions
)

print(f"adaboost Accuracy: {boost_accuracy:.2f}")
print(f"adaboost F1 Score: {boost_f1:.2f}")

#adaboost correct predictions
adaboost_correct = (y_test == boost_predictions).sum()
print("AdaBoost correct predictions:", adaboost_correct)

#ada boost confusion matrix
cm_boost = confusion_matrix(
    y_test,
    boost_predictions
)

display_boost = ConfusionMatrixDisplay(
    confusion_matrix=cm_boost,
    display_labels=["No", "Yes"]
)

display_boost.plot()
plt.title("AdaBoost Confusion Matrix")
plt.show()

#adaboost feature importance
feature_importance = pd.Series(
    boosting_model.feature_importances_,
    index=x.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print(feature_importance)

#adaboost feature importance graph
plt.figure(figsize=(10, 6))

feature_importance.plot(kind="barh")

plt.title("AdaBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


#adaboost cross validation
boost_cv_accuracy = cross_val_score(
    boosting_model,
    X_train,
    y_train,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

print("AdaBoost CV Scores:", boost_cv_accuracy)

print("AdaBoost Mean Accuracy:",
      boost_cv_accuracy.mean())

print("AdaBoost Standard Deviation:",
      boost_cv_accuracy.std())


#compare decision tree vs adaboost
print("Decision Tree CV Accuracy:",
      dt_cv_accuracy.mean())

print("AdaBoost CV Accuracy:",
      boost_cv_accuracy.mean())


'''AdaBoost was selected as the advanced machine learning technique because it is an ensemble boosting method that combines multiple weak learners. 
Each subsequent learner focuses on correcting the errors made by previous learners,
 which can improve classification performance compared with a single Decision Tree model.'''''

'''''The Decision Tree was selected as a conventional classification technique due to its interpretability, while Random Forest was selected as an advanced ensemble method that improves performance by combining predictions from multiple decision trees through bagging.
Decision Tree + AdaBoost
The Decision Tree was selected as a baseline classifier, while AdaBoost was selected as an advanced boosting technique that improves prediction performance by focusing on observations incorrectly classified by previous learners.'''''