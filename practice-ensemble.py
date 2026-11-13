import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

#load dataset
data= pd.read_csv("Practice_Data.csv")

#explore data
print(data.head())
print(data.shape) #rows and columns
print(data.info())# column names and data types
print(data.describe())  #stats info
print(data.isnull().sum()) #missing values
print(data.duplicated().sum())

print(data["price_range"].value_counts()) #count how many phones in each category to see if its balanced
#u need to check whether target variable i balanced for categorical target use "value_counts() for numerical tager use describe(). and a histogram maybe to understand its distribution
#to see percentages rather than just counts:
print(data["price_range"].value_counts(normalize=True) * 100)
#if dataset is imbalanced consider using classweight="balanced" when modeling

#then seperate x input features and y the target
x = data.drop("price_range", axis=1) #axis 1 drops a column rather than row
y = data["price_range"]

#if the target is categorical i would need to do label encoder, but here its not
#then split the dataset training 80% and testing 20%

X_train, X_test, y_train, y_test = train_test_split(x , y ,
                                                    test_size= 0.2,
                                                    random_state=1,
                                                    stratify=y)

print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])
#Why do we split the dataset?
#We split the dataset into training and testing data to evaluate how well the model performs on unseen data.
# This helps us understand whether the model has learned useful patterns rather than simply memorising the training data. It also helps identify overfitting.

#first traditional model: decision tree
decision_tree= DecisionTreeClassifier( random_state=1)
decision_tree.fit(X_train, y_train)
decision_predictions = decision_tree.predict(X_test)

decision_accuracy = accuracy_score(
    y_test,
    decision_predictions
)

decision_f1 = f1_score(
    y_test,
    decision_predictions,
    average="weighted"
)
# i have four categories , therefore i need to do average_weighted for both models because it calculates the F1 score for each category and combines them whole accounting for how many examples belong to each category
# if i had binary only two categories then no need for average.

print(f"Decision Tree Accuracy: {decision_accuracy:.2f}")
print(f"Decision Tree F1 Score: {decision_f1:.2f}")




#random forest bagging model

bagging_model = RandomForestClassifier(
    n_estimators=200,
    random_state=1,
    n_jobs=-1
) # max depth 1 restricts every decision tree in your Random Forest to just one level of decisions.
#That means each tree has very limited information to work with. Even though Random Forest combines 200 trees, those individual trees might be too simple to capture the patterns in the phone specifications.
bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(
    y_test,
    bagging_predictions
)
bagging_f1 = f1_score(
    y_test,
    bagging_predictions,
    average="weighted"
)
print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")# this suggests that Random Forest provided a better balance between precision and recall across the four price categories.

# Calculate how many predictions each model got correct
decision_correct = (y_test == decision_predictions).sum()
random_forest_correct = (y_test == bagging_predictions).sum()

print("Decision Tree correct predictions:", decision_correct)
print("Random Forest correct predictions:", random_forest_correct)

#Random Forest can be more reliable than a single Decision Tree because it combines predictions from multiple trees, reducing the impact of errors made by individual trees.
#next a confusion matrix to tell us which categories the model is predicting correctly and which categories it is confusing with one another.

#deicison tree confusion matrix
cm_dt = confusion_matrix(y_test, decision_predictions)

display_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
)

display_dt.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

#random forest confusion matrix

cm_rf = confusion_matrix(y_test, bagging_predictions)

display_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
)

display_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

#This is useful because it tells us that Random Forest didn't just achieve a higher overall accuracy. It also made more correct predictions for every individual price category.

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

#Cross-validation is NOT designed to improve the model's accuracy. It is designed to give us a better understanding of how reliably the model performs.