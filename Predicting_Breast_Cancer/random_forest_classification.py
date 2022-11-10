# Random Forest Classification

# Importing the libraries
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set
classifier = RandomForestClassifier(
    n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# type = dataset["Class"].value_counts()
# index_ = type.index
# values_ = type.values

# figure = px.pie(dataset,
#                 values=values_,
#                 names=index_, hole=0.5,
#                 title="Benign or Malignant")
# figure.show()

# fig = px.bar(dataset, x="nation", y="count", color="medal", title="Long-Form Input")
# fig.show()

dataset.hist(figsize=(30, 20))
plt.show()
