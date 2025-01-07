from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# KNN Classifier function
def KNNClassifier(x_train, y_train, x_test, k):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    return y_pred

# SVM Classifier function (w/ Kernels)
def SVMClassifier(x_train,y_train, x_test, params):
    # Set default values for parameters
    params.setdefault('C', 1)
    params.setdefault('gamma', 1)
    params.setdefault('kernel', 'linear')
    
    model = svm.SVC(C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    return y_pred