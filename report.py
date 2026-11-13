
"""
=========================================================
REPORT 1: CLASSIFICATION
CASE STUDY: MOBILE PHONE PRICE PREDICTION
MODELS: DECISION TREE VS RANDOM FOREST
=========================================================

1. INTRODUCTION AND DATA PRESENTATION

The aim of this study was to analyse how mobile phone
hardware specifications influence their market value.
This is a supervised classification problem, as the
objective is to predict the price range of a mobile
phone based on its hardware specifications.

The dataset contains 2,000 observations and 20 input
features, alongside one target variable called
'price_range'. The target variable is divided into
four categories: Low Cost, Medium Cost, High Cost and
Very High Cost. The aim was to build and evaluate two
classification models to determine which technique
provided the strongest predictive performance.


=========================================================
2. DATA PRE-PROCESSING
=========================================================

Before building the models, the dataset was inspected
to identify any data quality issues and determine
whether preprocessing was required.

Missing values and duplicate records were checked using
isnull().sum() and duplicated().sum(). It was found that
the dataset contained no missing values or duplicate
records.

The data types were also examined to ensure that the
variables were in a suitable format for modelling.
No changes were required, as all input features were
numerical.

No additional preprocessing, such as encoding or
scaling, was required because the dataset was already
in a suitable format for Decision Tree and Random
Forest modelling.

The target variable, 'price_range', was separated from
the input features. The dataset was then divided into
training and testing sets using an 80% / 20% split.
This allowed the models to learn patterns from the
training data and evaluate their performance on
unseen observations.


=========================================================
3. TECHNIQUE 1: DECISION TREE CLASSIFIER
=========================================================

The first traditional model selected was the Decision
Tree Classifier. This is a supervised machine learning
technique suitable for classification problems.

A Decision Tree works by using a series of decision
rules to split the data into smaller groups, based on
the input features, until it makes a prediction about
the target category.

The model was trained using the training dataset to
predict the mobile phone price range.

The random_state parameter was set to 1 to ensure
reproducible results. The maximum tree depth was left
at its default setting (None), allowing the tree to
grow without a specified depth limit.


=========================================================
4. TECHNIQUE 2: RANDOM FOREST CLASSIFIER
=========================================================

The second model selected was the Random Forest
Classifier, an ensemble learning technique that
combines multiple Decision Trees to make predictions.

Random Forest uses bagging, which involves training
multiple decision trees on different samples of the
training data. Their predictions are then combined
to produce a final prediction. This helps reduce the
impact of errors made by individual trees and can
improve predictive performance.

The model was configured with 200 trees
(n_estimators=200), with random_state set to 1 to
ensure reproducible results. The maximum depth was
also left at its default setting (None).


=========================================================
5. COMPARISON OF MODEL PERFORMANCE
=========================================================

Both models were evaluated using accuracy and weighted
F1-score on the unseen testing dataset, which contained
400 observations.

The Decision Tree achieved an accuracy of 82.5% and
a weighted F1-score of 0.82. In comparison, the Random
Forest achieved a higher accuracy of 89.5% and a
weighted F1-score of 0.89.

The confusion matrices also showed that Random Forest
correctly classified 358 mobile phones, compared with
330 for Decision Tree. Random Forest also achieved
more correct predictions across all four price
categories, indicating stronger classification
performance.

Feature importance was also examined to identify which
input features contributed most to the Random Forest
predictions. RAM had the highest importance score of
0.483, followed by battery power with a score of
0.0738. This suggests that RAM was the most influential
feature when predicting mobile phone price ranges.

Additionally, repeated stratified 10-fold cross-
validation was performed, with the process repeated
three times, to assess the consistency of both models
across different training and validation splits.

The Random Forest achieved a higher mean accuracy of
87.75%, compared with 83.67% for the Decision Tree.
It also produced a lower standard deviation of 0.0275,
compared with 0.0330 for the Decision Tree. This
suggests that Random Forest provided slightly more
consistent performance across the validation splits.

Overall, both the test results and cross-validation
results showed stronger performance from the Random
Forest model.


=========================================================
6. SUMMARY AND CONCLUSION
=========================================================

Overall, Random Forest achieved stronger performance
compared with the Decision Tree Classifier. It produced
a higher test accuracy and weighted F1-score, correctly
classified more mobile phones and achieved a higher
mean accuracy during cross-validation.

Therefore, Random Forest was considered the more
suitable technique for predicting mobile phone price
categories in this case study.


=========================================================
END OF CLASSIFICATION REPORT
=========================================================
"""

"""
# Customer Segmentation Using K-Means and Fuzzy C-Means

## 1. Introduction and Dataset

The aim of this study was to identify different customer groups using clustering techniques. Customer segmentation can help businesses understand customer behaviour and develop more targeted marketing strategies.

The Mall Customers dataset contained 200 observations and five variables: CustomerID, Gender, Age, Annual Income and Spending Score. As the dataset did not contain a target variable, this was an unsupervised machine learning problem. Two clustering techniques were applied: traditional K-Means and advanced Fuzzy C-Means (FCM).

## 2. Data Preprocessing

The dataset was first inspected to understand its structure, data types and missing values. There were 200 rows and five columns, with no missing values.

CustomerID was excluded because it is an identifier rather than a meaningful customer characteristic. Gender was also excluded, leaving Age, Annual Income and Spending Score as the three numerical features used for clustering.

StandardScaler was applied to standardise the features before clustering. This was important because clustering algorithms use distances to group observations, and features with larger numerical ranges could otherwise have a greater influence on the results. The scaled data had approximately zero means and standard deviations of one.

## 3. Technique 1: K-Means Clustering

K-Means was used as the traditional clustering technique. Different numbers of clusters (K=2 to K=10) were investigated using the elbow and silhouette methods.

The elbow plot suggested that four or five clusters could be suitable, while the silhouette plot showed its highest score at K=6. However, K=5 was selected as a reasonable balance between the two methods.

The final K-Means model identified five customer segments. The cluster profiles showed differences in customer characteristics.

K-Means cluster profiles:

Cluster 0:
- Average age: 46.2
- Average annual income: 26.8k
- Average spending score: 18.4
- Number of customers: 20

Cluster 1:
- Average age: 25.2
- Average annual income: 41.1k
- Average spending score: 62.2
- Number of customers: 54

Cluster 2:
- Average age: 32.9
- Average annual income: 86.1k
- Average spending score: 81.5
- Number of customers: 40

Cluster 3:
- Average age: 39.9
- Average annual income: 86.1k
- Average spending score: 19.4
- Number of customers: 39

Cluster 4:
- Average age: 55.6
- Average annual income: 54.4k
- Average spending score: 48.9
- Number of customers: 47

These results show that Cluster 0 represents lower-income, lower-spending customers. Cluster 2 represents high-income, high-spending customers, while Cluster 3 represents high-income customers with low spending scores. Cluster 1 contains younger customers with relatively lower incomes but higher spending scores. Cluster 4 contains older customers with more moderate income and spending scores.

The scatter plot showed that Clusters 0, 2 and 3 were relatively distinct, while Clusters 1 and 4 showed some overlap.

## 4. Technique 2: Fuzzy C-Means

FCM was used as the advanced clustering technique. Unlike K-Means, which assigns each customer to one cluster, FCM gives every customer a membership score for each cluster. This allows a customer to partially belong to multiple clusters.

The same five clusters and scaled features were used to make the comparison fair.

The membership matrix had a shape of (200, 5). Each row represented one customer, and the five membership values added up to approximately one. Provisional cluster labels were created using each customer's highest membership score.

FCM cluster profiles:

Cluster 0:
- Average age: 24.3
- Average annual income: 39.4k
- Average spending score: 63.7

Cluster 1:
- Average age: 62.7
- Average annual income: 49.8k
- Average spending score: 44.0

Cluster 2:
- Average age: 32.6
- Average annual income: 86.0k
- Average spending score: 81.5

Cluster 3:
- Average age: 39.9
- Average annual income: 87.7k
- Average spending score: 17.7

Cluster 4:
- Average age: 43.6
- Average annual income: 46.4k
- Average spending score: 38.4

The FCM profiles show that Cluster 2 represents high-income, high-spending customers, while Cluster 3 represents high-income, low-spending customers. These groups are similar to the corresponding K-Means segments.

The membership histogram showed variation in how strongly customers belonged to their assigned clusters. Three customers had particularly low maximum membership scores, between approximately 0.27 and 0.30. This indicates that their cluster assignments were less certain because their membership was more evenly distributed across different clusters.

## 5. Comparison of the Models

The cross-tabulation was used to compare the provisional FCM labels with the K-Means cluster assignments.

Cross-tabulation results:

                 FCM Cluster
K-Means       0    1    2    3    4
Cluster 0     0    4    0    0   16
Cluster 1    48    0    1    0    5
Cluster 2     0    0   39    0    1
Cluster 3     0    0    0   36    3
Cluster 4     0   25    0    0   22

The cross-tabulation showed similarities between the two techniques. For example, 39 customers from K-Means Cluster 2 were also assigned to FCM Cluster 2, while 36 customers from K-Means Cluster 3 were assigned to FCM Cluster 3.

However, K-Means Cluster 4 was divided mainly between FCM Clusters 1 and 4. K-Means Cluster 0 was also mainly assigned to FCM Cluster 4. This shows that the algorithms did not always group customers identically.

Overall, K-Means provided clear and straightforward assignments, while FCM offered additional information about uncertainty and overlapping customer behaviour through membership scores.

## 6. Conclusion

Both techniques identified meaningful customer segments based on age, annual income and spending score. K-Means provided a clear baseline for customer segmentation, while FCM offered a more flexible approach by showing the degree to which customers belonged to different groups.

FCM is particularly useful when customer behaviour overlaps between segments because it provides membership scores rather than only one fixed assignment. Both techniques can help businesses understand their customers and support more targeted marketing decisions.
"""
