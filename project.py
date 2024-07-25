# Mall Customer Segmentation

# Data collection
import pandas as pd

#Load the dataset
file_path="C:\Projects\CipherSchools Data Science Project\Mall_Customers.csv"
# converting the data into a dataframe
data=pd.read_csv(file_path)

#display the first few rows of the dataset
print(data.head(10))
count=data.isnull().sum()
print(count)
mean_age=data["Age"].mean()
data["Age"].fillna(mean_age,inplace=True)
print(data.head())
#changing the names of columns of data frame
data.columns=["CustomerID","Gender","Age","AnnualIncome","SpendingScore"]
print(data)
#data transformation i.e. categorical variables
mode_gender=data['Gender'].mode()[0]
count=data.isnull().sum()
print(count)
data["Gender"].fillna(mode_gender,inplace=True)
data["Gender"]=data["Gender"].map({'Male': 0, 'Female':1})
print(data.describe())
import matplotlib.pyplot as plt
import seaborn as sns
#histplot-Age,Annual income,Spending score
# Visualizing distributions
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['AnnualIncome'], bins=30, kde=True)
plt.title('Annual Income Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['SpendingScore'], bins=30, kde=True)
plt.title('Spending Score Distribution')
plt.show()

# Visualizing relationships
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='AnnualIncome', y='SpendingScore', hue='Gender')
plt.title('Income vs Spending Score')
plt.show()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Feature selection
features = data[['Age', 'AnnualIncome', 'SpendingScore']]

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluating cluster quality
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()