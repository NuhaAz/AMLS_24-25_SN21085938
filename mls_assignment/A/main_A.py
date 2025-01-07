import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2

# Import ML models
from models import *


def convert_to_high_contrast(image):
    _, bw_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    return bw_image

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
# Malignant: 0, Benign: 1

# Convert 2D array (image) into 1D array of pixel values for each data split
x_train = np.array([np.hstack(x) for x in data["train_images"]])
y_train = np.array(data["train_labels"]).ravel()
x_val = np.array([np.hstack(x) for x in data["val_images"]])
y_val = np.array(data["val_labels"]).ravel()
x_test = np.array([np.hstack(x) for x in data["test_images"]])
y_test = np.array(data["test_labels"]).ravel()

# Attempt at preprocessing data (not used)
# x_train_contrast = np.array([np.hstack(convert_to_high_contrast(x)) for x in data["train_images"]])
# x_val_contrast = np.array([np.hstack(convert_to_high_contrast(x)) for x in data["val_images"]])
# x_test_contrast = np.array([np.hstack(convert_to_high_contrast(x)) for x in data["test_images"]])
# print(x_train[1])
# print(x_train_contrast[1])
# exit()

model_sel = input("Model: ")
accuracy_all = []

if model_sel == "1":
    # KNN Classifier
    print("-------------KNN Classifier-------------")
    
    # Training and validation of KNN model for different K Values
    K = [i for i in range(1,int(np.sqrt(len(x_train))))]
    scores = []
    for k in K:
        y_pred = KNNClassifier(x_train, y_train, x_val, k)
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
    print("Best K Value: {}  |  Accuracy: {}".format(k, acccuracy))
    
    # Create confusion matrix for test set
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Malignant (True)', 'Benign (True)'], 
                                    index=['Malignant (Predict)', 'Benign (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()

elif model_sel == "2":
    # Linear SVM Classifier
    print("-------------Linear SVM Classifier-------------")
    C_vals = [1e-1, 5e-1, 1, 5, 1e2]
    
    # Training and Validation of SVM model
    # Validating for various C values
    scores = []
    for C in C_vals:
        params = {'C': C}
        y_pred = SVMClassifier(x_train, y_train, x_val, params)
        acccuracy = accuracy_score(y_val, y_pred)
        scores.append(acccuracy)
        # print("Params: {}  |  Accuracy: {}".format(params, acccuracy))
    
    # Plotting validation set accuracy scores for different C Values
    fig, ax = plt.subplots()
    plt.plot(C_vals, scores)
    plt.xlabel("C Value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs C Value - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('accuracy.png')
    plt.clf()

    # Test set results based on C with highest accuracy (found during validation)
    index = scores.index(max(scores))
    param = {'C': C_vals[index]}
    y_pred = SVMClassifier(x_train, y_train, x_test, param)
    acccuracy = accuracy_score(y_test, y_pred)
    accuracy_all.append(acccuracy)
    print("Best Params: {}  |  Accuracy: {}".format(param, acccuracy))
    
    # Create confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Malignant (True)', 'Benign (True)'], 
                                    index=['Malignant (Predict)', 'Benign (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()
    
elif model_sel == "3":    
    print("-------------SVM Classifier w/ RBF Kernel-------------")
    
    # Training and Validation of SVM model
    # Validating for various gamma values
    C = 1
    gamma_vals = [1e-1, 5e-1, 1, 5, 1e1]
    scores = []
    for gamma in gamma_vals:
        params = {'C': C, 'gamma': gamma, 'kernel': 'rbf'}
        y_pred = SVMClassifier(x_train, y_train, x_val, params)
        acccuracy = accuracy_score(y_val, y_pred)
        scores.append(acccuracy)
            
    # Plotting validation set accuracy scores for different gamma Values
    fig, ax = plt.subplots()
    plt.plot(gamma_vals, scores)
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs gamma - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('accuracy.png')
    plt.clf()

    # Test set results based on gamma with highest accuracy (found during validation)
    index = scores.index(max(scores))
    param = {'C': C, 'gamma': gamma_vals[index], 'kernel': 'rbf'}
    y_pred = SVMClassifier(x_train, y_train, x_test, param)
    acccuracy = accuracy_score(y_test, y_pred)
    accuracy_all.append(acccuracy)
    print("Best Params: {}  |  Accuracy: {}".format(param, acccuracy))
    
    # Create confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Malignant (True)', 'Benign (True)'], 
                                    index=['Malignant (Predict)', 'Benign (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()