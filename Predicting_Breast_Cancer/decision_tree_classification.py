# Decision Tree Classification

# Importing the libraries
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    r'D:/VS Code/Workspace/Predicting_Breast_Cancer/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#Missing Or Null data points
dataset.isnull().sum()
dataset.isna().sum()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision Tree Classification model on the Training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
CM_display = ConfusionMatrixDisplay(cm).plot()
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# check correlation
data_num = dataset.drop('Class', axis=1)
corr = data_num.corr()
print(corr)

# correlation map
f, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(data_num.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.figure(figsize=(10, 10))
sns.regplot(y_test, y_pred, fit_reg=True)

sns.set(style="darkgrid")
target = 'Class'
ax = sns.countplot(x=target, data=dataset)
print(dataset[target].value_counts())

CM_display = ConfusionMatrixDisplay(cm).plot()
