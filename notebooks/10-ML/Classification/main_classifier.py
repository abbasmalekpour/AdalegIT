import matplotlib.pyplot as plt
# Importing the libraries
import numpy as np
import pandas as pd
# Visualising the Test set results
# Visualising the Training set results
from matplotlib.colors import ListedColormap
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
import classifier




file_path = './Section 16 - Logistic Regression/Python/Social_Network_Ads.csv'
X_feature_list = [2,3] # only Column number 2 and 3 are selected as Independent Variables
target = -1 # last colomn


X, y = classifier.read_X_y_from_path(file_path, X_feature_list, target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier_model = classifier.build_classifier('LogisticRegression', X_train, y_train)

y_pred = classifier.pred(classifier_model, X_test)

classifier.metrix_plot(classifier_model, X_test, y_test, simple_CM=True, color_confusion_metrix=True, ROC_AUC=True);
plt.show();
classifier.classification_plot(classifier_model, X_train, y_train);
plt.show();
classifier.classification_plot(classifier_model, X_test, y_test)
plt.show();
print(classifier.metrix_report(y_test, classifier.pred(classifier_model, X_test)))