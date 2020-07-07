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


def read_X_y_from_path(file_path, X_feature_list, target):
    '''recieve file_path for CSV file and return X for Independent Variables and y for Target or Dependent Variable '''
    # Importing the dataset
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, X_feature_list].values
    y = dataset.iloc[:, target].values
    return X, y

def scale_it(X):
    '''Using Standard Scaler, Scale any give dataset to (-1,1):
    from sklearn.preprocessing import StandardScaler'''
    from sklearn.preprocessing import StandardScaler
    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X

def build_classifier(classifier, X_train, y_train):
    '''
    Please pass one of the following classifier name as your Classification Model:
    
    LogisticRegression
    KNeighborsClassifier
    naive_bayes
    SVC
    DecisionTreeClassifier
    RandomForestClassifier
    
    
    For example:
    build_classifier(RandomForestClassifier)
    '''
    if 'LogisticRegression'==classifier:
        classifier = LogisticRegression(random_state = 0)
        print(f'the {classifier} model has been built!!!\n')
        
    elif 'KNeighborsClassifier'==classifier:
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        print(f'the {classifier} model has been built!!!\n')
    
    elif 'naive_bayes'==classifier:
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        print(f'the {classifier} model has been built!!!\n')
    
    elif 'SVC'==classifier:
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        print(f'the {classifier} model has been built!!!\n')
    
    elif 'DecisionTreeClassifier'==classifier:
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        print(f'the {classifier} model has been built!!!\n')
    
    elif 'RandomForestClassifier'==classifier:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        print(f'the {classifier} model has been built!!!\n')
    
    
    classifier.fit(scale_it(X_train), y_train)
    return classifier

def pred(classifier, X_test):
    '''Predicting the given (Test) set results'''
    
    return classifier.predict(scale_it(X_test))

def metrix_plot(classifier, X_test, y_test, cmap='Greens', simple_CM=False, color_confusion_metrix=False, ROC_AUC=False):
    '''Choose to Plot between Simple Confustion Metrix or Colored Confusion Metrix or ROC_AUC'''
    X_test = scale_it(X_test)
    from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve
    
    # Making the Confusion Matrix
    if simple_CM == True:
        cm = confusion_matrix(y_test, pred(classifier, X_test))
        print(f'Confusion Metrix Results for {classifier} Model:\n', cm)

    # Making the Color Confusion Matrix
    if color_confusion_metrix==True:
        plot_confusion_matrix(classifier, X_test, y_test,cmap='Greens')

    #ROC_AUC
    if ROC_AUC==True:
        plot_roc_curve(classifier,X_test, y_test)
#         plt.show();
def metrix_report(y_test, y_pred):
    from sklearn.metrics import classification_report
    target_names = ['class 0', 'class 1']
    return classification_report(y_test, y_pred, target_names=target_names)

def classification_plot (classifier, X_train, y_train, X_lable='Age', y_label='Estimated Salary', title='Logistic Regression'):
    '''Plot a beutiful scatter plot for two class Classifier data with Green(1) and Red(0)'''
    
    plt.rcParams['figure.figsize'] = [15,6] 
    
    X_set, y_set = scale_it(X_train), y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.2, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel(X_lable)
    plt.ylabel(y_label)
    plt.legend()
    plt.show();
    


# print(scale_it(y_test))
# print(X_train[0],' Become:', scale_it(X_train)[0])
