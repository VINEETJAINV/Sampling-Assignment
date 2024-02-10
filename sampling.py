import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Datasets/Creditcard_data.csv')
df.head()

from sklearn.preprocessing import normalize
Amount = normalize([df['Amount']])[0]
df['Amount'] = Amount
df = df.iloc[:, 1:]
df.head()

class_freq = df['Class'].value_counts()
print(f"Number of classes: {len(class_freq)}")
print("Class frequencies:")
print(class_freq)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Instantiate oversampler and undersampler
oversampler = RandomOverSampler()
undersampler = RandomUnderSampler()

# Resample using both oversampling and undersampling
X_over, y_over = oversampler.fit_resample(X, y)
X_resampled, y_resampled = undersampler.fit_resample(X_over, y_over)

# Print the number of samples in each class
print("Number of samples in each class after resampling:")
print(y_resampled.value_counts())

X_resampled

new_df = pd.concat([X_resampled, y_resampled], axis=1)
new_df

class_freq = new_df['Class'].value_counts()
print(f"Number of classes: {len(class_freq)}")
print("Class frequencies:")
print(class_freq)

# simple random sampling
n = int((1.96*1.96 * 0.5*0.5)/(0.05**2))
sampled_df = new_df.sample(n=n, random_state=42)
print(sampled_df)

# systematic sampling

interval = 2
systematic_df = new_df.iloc[::interval]
systematic_df

# stratified sampling

from sklearn.model_selection import train_test_split

n = int((1.96*1.96 * 0.5*0.5)/((0.05)**2))

strata = new_df.groupby('Class')

# sample 2 rows from each stratum
stratified_df = strata.apply(lambda x: x.sample(n))

stratified_df

# cluster sampling
import numpy as np

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42).fit(new_df)
cluster_assignments = kmeans.labels_

# Select the clusters you want to include in the sample
selected_clusters = [0, 2, 4, 5, 8]

cluster_series = pd.Series(cluster_assignments)

# Create the new DataFrame containing only the rows from the selected clusters
df_cluster_sample = new_df[cluster_series.isin(selected_clusters)]


# Print the resulting DataFrame
(df_cluster_sample)

# convenience sampling
convenience_sample = pd.concat([new_df.head(380), new_df.tail(380)])
convenience_sample

!pip install pycaret

from pycaret.classification import *
# !pip uninstall scikit-learn
# !pip install scikit-learn==0.23.2

setup(data = sampled_df, target='Class', silent=True)
cm = compare_models()

from pycaret.classification import *

setup(data = systematic_df, target='Class', silent=True)
cm = compare_models()

setup(data = stratified_df, target='Class', silent=True)
cm = compare_models()

setup(data = df_cluster_sample, target='Class', silent=True)
cm = compare_models()

setup(data = convenience_sample, target='Class', silent=True)
cm = compare_models()