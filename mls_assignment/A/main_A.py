import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Import ML models
from models import KNNClassifier, SVMClassifier 


# Load data
file_loc = "../datasets/breastmnist.npz"
data = np.load(file_loc)
# Data          | Shape
#----------------------------
# train_images  | (546x28x28)
# train_labels  | (546x1)
# val_images    | (78x28x28)
# val_labels    | (78x1)
# test_images   | (156x28x28)
# test_labels   | (156x1)
#----------------------------
# Benign: 0, Malignant: 1

# Convert 2D array (image) into 1D array of pixel values for each data split
x_train = np.array([np.hstack(x) for x in data["train_images"]])
y_train = np.array(data["train_labels"]).ravel()
x_val = np.array([np.hstack(x) for x in data["val_images"]])
y_val = np.array(data["val_labels"]).ravel()
x_test = np.array([np.hstack(x) for x in data["test_images"]])
y_test = np.array(data["test_labels"]).ravel()

model_sel = input("Model: ")
accuracy_all = []

if model_sel == "1":
    # KNN Classifier
    print("-------------KNN Classifier-------------")
    
    # Training and validation of KNN model for different K Values
    K = [i for i in range(1,int(np.sqrt(len(x_train))))]
    scores = []
    preds = []
    for k in K:
        y_pred = KNNClassifier(x_train, y_train, x_val, k)
        preds.append(y_pred)
        scores.append(accuracy_score(y_val, y_pred))

    # Plotting validation set accuracy scores for different K Values
    fig, ax = plt.subplots()
    plt.plot(K, scores)
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K Value - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('accuracy.png')
    plt.clf()

    # Test set results based on k with highest accuracy (found during validation)
    index = scores.index(max(scores))
    k = K[index]
    y_pred = KNNClassifier(x_train, y_train, x_test, k)
    acccuracy = accuracy_score(y_test, y_pred)
    accuracy_all.append(acccuracy)
    print("K Value: {}  |  Accuracy: {}".format(k, acccuracy))
    
    # Create confusion matrix for test set
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Benign (True)', 'Malignant (True)'], 
                                    index=['Benign (Predict)', 'Malignant (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()

elif model_sel == "2":
    # SVM Classifier w/o Kernels
    print("-------------SVM Classifier w/o Kernels-------------")
    
    # Training of SVM model
    y_pred = SVMClassifier(x_train, y_train, x_test)
    acccuracy = accuracy_score(y_test, y_pred)
    accuracy_all.append(acccuracy)
    print("Accuracy: {}".format(acccuracy))
    
    # Create confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Benign (True)', 'Malignant (True)'], 
                                    index=['Benign (Predict)', 'Malignant (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()