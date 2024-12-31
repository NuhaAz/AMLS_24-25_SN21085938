from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# KNN Classifier function
def KNNClassifier(x_train, y_train, x_test, k):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    return y_pred

# SVM Classifier function (w/o Kernels)
def SVMClassifier(x_train,y_train, x_test):
    model = svm.SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    return y_pred