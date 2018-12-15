# Final Project Machine Learning (HCMUTE) - Using SVM & PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('arrhythmia_data - Copy.xlsx')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#=============================================================
"""from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
results = []
i = 151
while i >= 0:
    pca = PCA(n_components=i)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train) # XÂy model
    y_pred_test = classifier.predict(X_test)
    if accuracy_score(y_test, y_pred_test) > 0.75:
        results.append(i)
    i-=1"""

#=============================================================
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Plotting the Cumulative Summation of the Explained Variance
"""plt.figure()
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()"""
    

#=====================================================
# Fitting logistic Regresstion to the Training set
from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_test)
print('accuracy_score_train= ', accuracy_score(y_train, y_pred_train) * 100)
print('accuracy_score_test= ', accuracy_score(y_test, y_pred_test) * 100)