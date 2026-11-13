
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from fcmeans import FCM
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
data = pd.read_csv("Wholesale customers data.csv", encoding="latin1")


#explore data
print(data.head())
print(data.shape) #rows and columns
print(data.info())# column names and data types
print(data.describe())  #stats info
print(data.isnull().sum()) #missing values
print(data.duplicated().sum())



#select the numerical spending features only and scale them because k means uses distance to group customers , without scaling columns with much larger values would have too much influence on the clsuters
#These are actual spending amounts. region and chanel are not spenign behavior. and we are trying to group customers based on their spending behaviour
#Region and Channel were excluded from clustering because the objective was to segment customers based on purchasing behaviour rather than geographic or business-type classifications.
features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

#ceck for outliers
plt.figure(figsize=(12, 6))

sns.boxplot(data=data[features])

plt.title("Boxplots of Wholesale Customer Features")
plt.xticks(rotation=45)

plt.show()

'''''Boxplot analysis revealed the presence of outliers in several spending variables. These values were retained because they are likely to represent genuine high-spending customers rather than data entry errors. 
Removing them could reduce the ability of the clustering algorithms to identify important customer segments.'''''

X = data[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #Feature scaling was applied using StandardScaler to ensure all variables contributed equally to distance calculations and cluster formation.

print("Scaled shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3))


#test different numbers of clusters , choose k using evidence
ks = range(2, 11)  #Test different numbers of clusters, from 2 to 10
inertias, silhouettes = [], []

#Inertia measures how close customers are to the centre of their assigned cluster.
#Lower inertia means customers are closer to their cluster centres.
#Inertia will generally decrease as we increase K.
#We look for the point where the decrease starts slowing down. This is the elbow.

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Create two plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Elbow plot
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")

# Silhouette plot
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")

plt.tight_layout()
plt.show()

print(pd.DataFrame({
    "K": list(ks),
    "inertia": inertias,
    "silhouette": silhouettes
}).round(3))

#silhoutte score graph
#This measures how well-separated the clusters are.
#Closer to +1 = clusters are generally well separated.
#Around 0 = clusters overlap.
#Below 0 = some customers may have been assigned to the wrong cluster.
#elbow method measure inertia
'''''The Elbow Method suggested a possible range between 4 and 6 clusters, however the Silhouette Score achieved its highest value at K=3 (0.549). 
Since Silhouette Analysis directly evaluates cluster cohesion and separation, K=3 was selected as the optimal number of clusters.
I'd usually trust the silhouette score more because it is an objective clustering quality metric.'''''

#build k means clustering model (traditional) K-Means: Each customer belongs to one cluster.

# Choose the number of clusters
k = 3

# Create and fit the K-Means model
kmeans = KMeans(
    n_clusters=k,    # Number of customer groups; chosen using elbow and silhouette evidence
    random_state=42, # Makes the results reproducible when running the code again
    n_init=10         # Runs K-Means 10 times with different starting centroids to find a good solution
)
data["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, data["kmeans_cluster"])
print("K-Means silhouette score:", round(kmeans_silhouette, 3)) #A score closer to 1 generally means better-separated clusters. A score around 0 means clusters overlap, and a negative score can suggest customers have been placed in the wrong clusters.


#profiel the clusters, to understand wwhat makes each customer group different

# Calculate the average spending for each cluster
cluster_means=data.groupby("kmeans_cluster")[features].mean().round(1)
print(cluster_means.T) #the t stands fro transpose to see all the features and there clusters

#calculates the average spending across the six product categories for each cluster.
#For example, we might discover:Cluster 0 spends more on Fresh products.Cluster 1 spends more on Grocery and Milk. Cluster 2 spends less across most categories.
#These are examples of what we might find, not your actual results yet.


# Count how many customers are in each cluster
print(data["kmeans_cluster"].value_counts().sort_index())

#the fact that theres only 2 people in cluster 2 tells us we are seeing the results of the outliers , k means tells us that These customers are so different that they deserve their own cluster."
#or potentially explore k=2

#create a scatter plot
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=data,
    x=features[1],
    y=features[2], #these features can be selected and justified by lookign at the code for the k means cluster and how each feature is divided into the clusters. the more the variation the better to use as a feature
    hue="kmeans_cluster",
    palette="tab10"
)

plt.title("K-means customer segments")
plt.tight_layout()
plt.show()

'''''Cluster 1 (orange) contains the majority of customers and exhibits the lowest Milk and Grocery expenditure. This suggests a segment of lower-spending or smaller wholesale customers.
Cluster 0 (blue) demonstrates substantially higher Milk and Grocery expenditure than Cluster 1, indicating medium-to-large volume customers with greater purchasing activity.
Cluster 2 (green) contains only two customers but exhibits exceptionally high Milk expenditure and relatively high Grocery expenditure. This suggests a small group of premium or high-value wholesale customers with purchasing behaviour distinct from the rest of the customer base.'''''


#fuzzy c means advanced model Fuzzy C-Means: Each customer can have a membership score for every cluster.
#Use the same six scaled features and the same K value so we can compare the two models fairly.

# Create the Fuzzy C-Means model
fcm = FCM(n_clusters=k, random_state=42)

# Fit the model using scaled data
fcm.fit(X_scaled)

# Get the membership values
U = fcm.u

# Get the cluster centres
centres_scaled = fcm.centers

# Check the membership matrix
print("Membership matrix shape:", U.shape)

# Display membership values for the first five customers
print("First five membership vectors:\n", np.round(U[:5], 3))

# Check that membership values add up to approximately 1
print("Row sums (should be about 1):", U.sum(axis=1)[:5].round(4))

# Assign each customer to their strongest cluster (provisional label)
data["fcm_cluster"] = U.argmax(axis=1)

# Find each customer's highest membership value
data["fcm_max_membership"] = U.max(axis=1)

# Calculate the difference between the two highest membership values
sorted_U = np.sort(U, axis=1)

data["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]


#compare different k values
#The purpose is to see how changing the number of clusters affects how clearly customers fit into them.

comparison = []

for candidate_k in [3, 4, 5, 6]:

    model = FCM(n_clusters=candidate_k, random_state=42)
    model.fit(X_scaled)

    U_k = model.u

    sorted_memberships = np.sort(U_k, axis=1)
    max_membership = U_k.max(axis=1)
    gap = sorted_memberships[:, -1] - sorted_memberships[:, -2]

    comparison.append({
        "K": candidate_k,
        "mean_max_membership": max_membership.mean(),
        "median_max_membership": np.median(max_membership),
        "mean_top_two_gap": gap.mean()
    })

#First, remember that we're comparing four different versions of Fuzzy C-Means, changing the number of clusters (K) each time.

print(pd.DataFrame(comparison).round(3))
#Measure: Mean maximum membership	- On average, how strongly customers belong to their strongest cluster
#Median maximum membership -	The middle customer's strongest membership score
#Mean top-two gap -	How much stronger the best cluster membership is compared with the second-best

'''''notice that as the k increases the mena max memebership decreases , so the lower k is equal to 3 is better '''

#Your results suggest:1. K=3 gives the strongest average membership. 2. K=3 gives the strongest typical membership. 3. K=3 gives the largest gap between the two strongest memberships.
#This suggests that three clusters produce a sharper fuzzy partition than four, five or six clusters.


#visualize and profile FCM
plot_df = data.copy()

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=plot_df,
    x=features[1],
    y=features[2],
    hue="fcm_cluster", #This colours each customer according to their strongest FCM membership.
    palette="tab10"
)

plt.title("FCM: provisional labels by maximum membership")
plt.tight_layout()
plt.show()


#membership sharpness histogram
#This shows how strongly customers belong to their highest-membership cluster.
#Values closer to 1 = customers have a strong membership in one cluster.
#Values closer to 0.2 (with five clusters) = memberships are more evenly spread.

plt.figure(figsize=(7, 4))

sns.histplot(data["fcm_max_membership"], bins=20, kde=True)

plt.xlabel("Maximum FCM membership")
plt.title("How sharply are customers assigned?")
plt.tight_layout()
plt.show()


#It calculates the average spending across all six product categories for each cluster.
#It helps us understand what makes each customer group different.
print(data.groupby("fcm_cluster")[features].mean().round(1))

'''''interpret the histogram : 
Looking at your histogram

The x-axis runs approximately from:

Plain Text
0.35 → 0.95

and many customers are concentrated around:

Plain Text
0.80 - 0.90

That's actually good.

It means:

Most customers have a strong affiliation with one dominant cluster.

If the histogram had most values around:

Plain Text
0.35 - 0.45

that would be bad, because it would mean:

Most customers don't clearly belong anywhere.'''

# ==========================================
# HIERARCHICAL CLUSTERING (ADVANCED MODEL)
# ==========================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Use the same K chosen from K-Means
k = 3

# Build Hierarchical Clustering Model
hier = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)

# Fit model and assign cluster labels
data["hierarchical_cluster"] = hier.fit_predict(
    X_scaled
)

# ==========================================
# EVALUATION
# ==========================================

# Silhouette Score
hier_silhouette = silhouette_score(
    X_scaled,
    data["hierarchical_cluster"]
)

print(
    "Hierarchical Silhouette Score:",
    round(hier_silhouette, 3)
)

# ==========================================
# CLUSTER PROFILES
# ==========================================

print(
    "\n--- HIERARCHICAL CLUSTER PROFILES ---"
)

print(
    data.groupby(
        "hierarchical_cluster"
    )[features].mean().round(1)
)

# Count customers in each cluster

print(
    "\nCustomers per Cluster:"
)

print(
    data["hierarchical_cluster"]
    .value_counts()
    .sort_index()
)

# ==========================================
# VISUALISATION
# ==========================================

plt.figure(figsize=(7,5))

sns.scatterplot(
    data=data,
    x=features[1], # Milk
    y=features[2], # Grocery
    hue="hierarchical_cluster",
    palette="tab10"
)

plt.title(
    "Hierarchical Clustering (Ward Linkage)"
)

plt.tight_layout()

plt.show()

# ==========================================
# COMPARE WITH K-MEANS
# ==========================================

print(
    "K-Means Silhouette Score:",
    round(kmeans_silhouette,3)
)

print(
    "Hierarchical Silhouette Score:",
    round(hier_silhouette,3)
)

#optional dendogram
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

Z = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(10,5))

dendrogram(Z)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# ==========================================
# HIERARCHICAL CLUSTERING NOTES
# ==========================================

# Hierarchical Clustering is an unsupervised
# machine learning algorithm used to group
# similar observations together.

# Unlike K-Means, Hierarchical Clustering
# does not use centroids.

# Instead, it repeatedly merges the closest
# clusters until the desired number of
# clusters is reached.

# Ward linkage merges clusters while
# minimizing within-cluster variance.

# Hierarchical Clustering creates a tree-like
# structure called a dendrogram.

# The dendrogram helps visualize how clusters
# are related and where natural groupings
# occur.

# ==========================================
# COMPARISON WITH K-MEANS
# ==========================================

# K-Means:
# - Centroid based
# - Fast and efficient
# - Requires predefined K
# - Produces hard clusters

# Hierarchical Clustering:
# - Tree based clustering
# - Uses a dendrogram
# - Shows relationships between clusters
# - Does not depend on centroid placement

# ==========================================
# BUSINESS INTERPRETATION
# ==========================================

# K-Means:
# Useful when clear customer segments are
# required for marketing or targeting.

# Hierarchical:
# Useful when understanding relationships
# between customer groups is important.

# ==========================================
# EVALUATION
# ==========================================

# Silhouette Score:
# Measures how well separated clusters are.

# Closer to +1:
# Better cluster separation

# Around 0:
# Clusters overlap

# Negative:
# Poor clustering

# Higher silhouette score indicates a better
# clustering solution.

# ==========================================
# EXAM SENTENCE
# ==========================================

# K-Means was selected as the traditional
# clustering technique due to its simplicity
# and computational efficiency.

# Hierarchical Clustering was selected as
# the advanced technique because it reveals
# relationships between customer groups using
# a dendrogram and does not rely on centroid
# initialization.

# ==========================================
# CONCLUSION TEMPLATE
# ==========================================

# Compare:
# - Silhouette Scores
# - Cluster Profiles
# - Cluster Sizes

# The model with the higher silhouette score
# generally provides better cluster
# separation.

# Hierarchical clustering additionally
# provides insight into how clusters are
# related through the dendrogram.