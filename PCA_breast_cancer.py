# import packages
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
breast = load_breast_cancer()
breast_data = breast.data
#print(breast_data)

# shape of the data
print(breast_data.shape)

# create labels
breast_labels = breast.target
# shape of the target
print(breast_labels.shape)

# reshaping the label dataset
labels = np.reshape(breast_labels, (569,1))
print(labels.shape)

# conacatenating in the final dataset (data with labels)
final_breast_data = np.concatenate([breast_data, labels], axis=1)

# checking the shape of the final data
print(final_breast_data.shape)

# print the feature names
features = breast.feature_names
print(features)

# creating a dataframe
df = pd.DataFrame(final_breast_data)
print(df.head())

# adding the features in the final dataframe
feature_labels = np.append(features, 'label')

# adding the features in dataframe
df.columns = feature_labels

# print the feature names
print(df.head())

# replacing with benign and malignant
df['label'].replace(0, 'Benign',inplace=True)
df['label'].replace(1, 'Malignant',inplace=True)

# print the tail
print(df.tail())

# standardscaler for pca
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features

# printing the shape
print(x.shape)

# checking the mean and std
print("mean: ", np.mean(x))
print("standard deviation: ", np.std(x))

# changing the features to a list
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
print(normalised_breast.head())

# time for PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

# creating a dataframe for the created PCA
principal_breast_Df = pd.DataFrame(data = principalComponents_breast, columns = ['principal component 1', 'principal component 2'])

# printing the tail
print(principal_breast_Df.tail())

# variation per principal component
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

# visualizing
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = df['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1'], principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()