import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline

# Load Dataset (Placeholder - Replace with actual dataset path)
df = pd.read_csv("customer_journey_data.csv")

# Initial Data Exploration
print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Handling Missing Data
imputer = SimpleImputer(strategy='median')
df_numeric = df.select_dtypes(include=[np.number])
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# Handling Categorical Data
categorical_cols = df.select_dtypes(include=['object']).columns
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
df_categorical = pd.DataFrame(ohe.fit_transform(df[categorical_cols]))
df_categorical.columns = ohe.get_feature_names_out(categorical_cols)

# Combine Processed Data
df_processed = pd.concat([df_numeric_imputed, df_categorical], axis=1)

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=['PCA1', 'PCA2'])

# Clustering - KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca)

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='viridis')
plt.title('Customer Segmentation using K-Means Clustering')
plt.show()

# Save Processed Data
df_pca.to_csv("processed_customer_journey_data.csv", index=False)