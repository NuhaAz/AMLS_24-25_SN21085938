import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
import cv2
import torch.optim as optim
from tqdm import tqdm

# Import ML models
from models import *
from utils import *

# Load dataset
print("Loading dataset...")
file_loc = "../datasets/breastmnist.npz"
x_train, y_train, x_val, y_val, x_test, y_test = loaddata(file_loc)

model_sel = input("Model: ")
precision_all = []

if model_sel == "1":
    # KNN Classifier
    print("-------------KNN Classifier-------------")
    
    # Training and validation of KNN model for different K Values
    K = [i for i in range(1,int(np.sqrt(len(x_train))))]
    scores = []
    for k in K:
        y_pred = KNNClassifier(x_train, y_train, x_val, k)
        scores.append(precision_score(y_val, y_pred))

    # Plotting validation set precision scores for different K Values
    fig, ax = plt.subplots()
    plt.plot(K, scores)
    plt.xlabel("K Value")
    plt.ylabel("Precision")
    plt.title("Precision vs K Value - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('precision.png')

    # Test set results based on k with highest precision (found during validation)
    index = scores.index(max(scores))
    k = K[index]
    y_pred = KNNClassifier(x_train, y_train, x_test, k)
    precision = precision_score(y_test, y_pred)
    precision_all.append(precision)
    recall = recall_score(y_test, y_pred)
    
    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    area = auc(fpr, tpr)
    
    # Print metrics
    print("Best K Value: {}  |  Precision: {}, Recall: {}, AUC: {}".format(k, precision, recall, area))
    
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
        precision = precision_score(y_val, y_pred)
        scores.append(precision)
    
    # Plotting validation set precision scores for different C Values
    fig, ax = plt.subplots()
    plt.plot(C_vals, scores)
    plt.xlabel("C Value")
    plt.ylabel("Precision")
    plt.title("Precision vs C Value - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('precision.png')

    # Test set results based on C with highest precision (found during validation)
    index = scores.index(max(scores))
    param = {'C': C_vals[index]}
    y_pred = SVMClassifier(x_train, y_train, x_test, param)
    precision = precision_score(y_test, y_pred)
    precision_all.append(precision)
    recall = recall_score(y_test, y_pred)
    
    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    area = auc(fpr, tpr)
    
    # Print metrics
    print("Best Params: {}  |  Precision: {}, Recall: {}, AUC: {}".format(param, precision, recall, area))
    
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
        precision = precision_score(y_val, y_pred)
        scores.append(precision)
            
    # Plotting validation set precision scores for different gamma Values
    fig, ax = plt.subplots()
    plt.plot(gamma_vals, scores)
    plt.xlabel("gamma")
    plt.ylabel("Precision")
    plt.title("Precision vs gamma - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('precision.png')

    # Test set results based on gamma with highest precision (found during validation)
    index = scores.index(max(scores))
    param = {'C': C, 'gamma': gamma_vals[index], 'kernel': 'rbf'}
    y_pred = SVMClassifier(x_train, y_train, x_test, param)
    precision = precision_score(y_test, y_pred)
    precision_all.append(precision)
    recall = recall_score(y_test, y_pred)
    
    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    area = auc(fpr, tpr)
    
    # Print metrics
    print("Best Params: {}  |  Precision: {}, Recall: {}, AUC: {}".format(param, precision, recall, area))
    
    # Create confusion matrix
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Malignant (True)', 'Benign (True)'], 
                                    index=['Malignant (Predict)', 'Benign (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()
    
elif model_sel == "4":
    print("-------------CNN-------------")
    
    # Hyperparameters
    num_epochs = 50
    batch_size = 16
    # lr = 0.005
    # lambda2 = 0.01
    dropout = 0.3
    lr = 0.01
    lambda2 = 0.005
    
    # Place data into train loaders for CNN inputs
    print('Preparing data...')
    train_loader = prepdata_cnn(x_train, y_train, batch_size)
    val_loader = prepdata_cnn(x_val, y_val, batch_size)
    test_loader = prepdata_cnn(x_test, y_test, batch_size)
    torch.backends.cudnn.benchmark = True
    
    # Create Model and training parameters
    print("Creating model...")
    model = CNN(dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lambda2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = Trainer(model, train_loader, criterion, optimizer, scheduler)
    validator = Validator(model, val_loader, criterion)
    
    # Settings for early stopping
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Other prerequisites
    start = 0
    epoch_list = []
    scores = []
    output="cnn_models"
    if not os.path.exists(output):
        os.makedirs(output)
    
    # Train and validation
    print("Beginning training...")
    print(f"Hyperparameters: batch size={batch_size}, learning rate={lr}, L2 lambda={lambda2}, dropout={dropout}")
    for epoch in tqdm(range(start,num_epochs)):
        epoch_list.append(epoch+1)
        train_metrics = trainer.train(epoch+1)
        val_metrics = validator.validate(epoch+1)
        
        scores.append(val_metrics[1])
        
        # Early stopping
        avg_val_loss = val_metrics[0]
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch = epoch+1
            torch.save(model.state_dict(), 'cnn_models/best_model.pth')
            
        else:
            epochs_no_improve += 1
            
        # Triggering early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered...")
            break
    
    # Plotting validation set precision scores for different epochs
    fig, ax = plt.subplots()
    plt.plot(epoch_list, scores)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision vs Epoch - BreastMNIST")
    plt.grid()  
    plt.show()
    fig.savefig('precision.png')

    # Test set results based on epoch with highest precision (found during validation)
    model.load_state_dict(torch.load('cnn_models/best_model.pth'))
    tester = Tester(model, test_loader, criterion)
    test_metrics = tester.test(threshold=0.5)
    y_pred = test_metrics[1]
    precision = precision_score(y_test, y_pred)
    precision_all.append(precision)
    recall = recall_score(y_test, y_pred)
    
    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    area = auc(fpr, tpr)
    
    # Print metrics
    print("Best Epoch: {}  |  Precision: {}, Recall: {}, AUC: {}".format(best_epoch, precision, recall, area))
    
    # Create confusion matrix for test set
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = pd.DataFrame(con_mat, columns=['Malignant (True)', 'Benign (True)'], 
                                    index=['Malignant (Predict)', 'Benign (Predict)'])

    fig, ax = plt.subplots()
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
    plt.show()
    fig.savefig('confusion_matrix.png')
    plt.clf()