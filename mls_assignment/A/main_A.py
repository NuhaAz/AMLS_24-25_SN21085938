import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, accuracy_score, ConfusionMatrixDisplay
import cv2
import torch.optim as optim
from tqdm import tqdm

# Import ML models
from models import *
from utils import *

print("\n")
print("{:-^50}".format(" Task A "))

# Load dataset
print("Loading BreastMNIST dataset...")
# file_loc = "../datasets/breastmnist.npz"
file_loc = "datasets/breastmnist.npz"
x_train, y_train, x_val, y_val, x_test, y_test = loaddata(file_loc)

# KNN Classifier
print("{:-^40}".format(" KNN Classifier "))

# Training and validation of KNN model for different K Values
K = [i for i in range(1,int(np.sqrt(len(x_train))))]
scores = []
for k in K:
    y_pred = KNNClassifier(x_train, y_train, x_val, k)
    scores.append(roc_auc_score(y_val, y_pred))

# # Plotting validation set auc scores for different K Values
# fig, ax = plt.subplots()
# plt.plot(K, scores)
# plt.xlabel("K Value")
# plt.ylabel("AUC")
# plt.title("AUC vs K Value - BreastMNIST")
# plt.legend(["Validation"])
# plt.grid()  
# plt.show()
# fig.savefig('knn_AUC.png')

# Test set results based on k with highest auc score (found during validation)
index = scores.index(max(scores))
k = K[index]
y_pred = KNNClassifier(x_train, y_train, x_test, k)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
area = roc_auc_score(y_test, y_pred)

# Print metrics
print("Best K Value: {}  |  Accuracy: {}, Precision: {}, Recall: {}, AUC: {}".format(k, accuracy, precision, recall, area))

# # Create confusion matrix for test set
# con_mat = confusion_matrix(y_test, y_pred)
# con_mat = pd.DataFrame(con_mat, columns=['Malignant (Predict)', 'Benign (Predict)'], 
#                                 index=['Malignant (True)', 'Benign (True)'])

# fig, ax = plt.subplots()
# sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
# plt.show()
# fig.savefig('knn_confusion_matrix.png')
# plt.clf()

# SVM Classifier w/ Polynomial Kernel
print("{:-^40}".format(" SVM w/ Polynomial Kernel "))

# Training and Validation of SVM model
# Validating for various polynomial degrees
C = 0.1
degree_vals = [x for x in range(1,11)]
scores = []
for degree in degree_vals:
    params = {'C': C, 'degree': degree, 'kernel': 'poly'}
    y_pred = SVMClassifier(x_train, y_train, x_val, params)
    scores.append(roc_auc_score(y_val, y_pred))
        
# # Plotting validation set AUC scores for different polynomial degrees
# fig, ax = plt.subplots()
# plt.plot(degree_vals, scores)
# plt.xlabel("Degree")
# plt.ylabel("AUC")
# plt.title("AUC vs Polynomial Degree - BreastMNIST")
# plt.legend(["Validation"])
# plt.grid()  
# plt.show()
# fig.savefig('svmpoly_AUC.png')

# Test set results based on degree with highest accuracy (found during validation)
index = scores.index(max(scores))
param = {'C': C, 'degree': degree_vals[index], 'kernel': 'poly'}
y_pred = SVMClassifier(x_train, y_train, x_test, param)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
area = roc_auc_score(y_test, y_pred)

# Print metrics
print("Best Params: {}  |  Accuracy: {}, Precision: {}, Recall: {}, AUC: {}".format(param, accuracy, precision, recall, area))

# # Create confusion matrix
# con_mat = confusion_matrix(y_test, y_pred)
# con_mat = pd.DataFrame(con_mat, columns=['Malignant (Predict)', 'Benign (Predict)'], 
#                                 index=['Malignant (True)', 'Benign (True)'])

# fig, ax = plt.subplots()
# sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
# plt.show()
# fig.savefig('svmpoly_confusion_matrix.png')
# plt.clf()
    
# CNN Classifier
print("{:-^40}".format(" CNN Classifier "))
    
# Hyperparameters
num_epochs = 100
batch_size = 16
lr = 0.008
lambda2 = 0.1
dropout = 0.3

# Place data into train loaders for CNN inputs
print('Preparing data...')
train_loader = prepdata_cnn(x_train, y_train, batch_size)
val_loader = prepdata_cnn(x_val, y_val, batch_size)
test_loader = prepdata_cnn(x_test, y_test, batch_size)
torch.backends.cudnn.benchmark = True

# Create Model and training parameters
print("Creating model...")
model = CNN(dropout=dropout)
model.init_weights()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lambda2)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

trainer = Trainer(model, train_loader, criterion, optimizer)
validator = Validator(model, val_loader, criterion)

# Settings for early stopping
patience = 5
best_auc = 0
epochs_no_improve = 0

# Other prerequisites
start = 0
epoch_list = []
scores = []
train_loss_all = []
val_loss_all = []
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
    
    auc = val_metrics[3]
    scores.append(auc)
    train_loss_all.append(train_metrics[0])
    val_loss_all.append(val_metrics[0])
    
    
    if epoch+1 >= 10:
        # Early stopping
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            best_epoch = epoch+1
            torch.save(model.state_dict(), 'cnn_models/best_model.pth')
            
        else:
            epochs_no_improve += 1
            
        # Triggering early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered...")
            break

# Plotting losses over all epochs
# fig, ax = plt.subplots()
# plt.plot(epoch_list, train_loss_all)
# plt.plot(epoch_list, val_loss_all)
# plt.xlabel("Epoch")
# plt.ylabel("Binary Cross Entropy Loss")
# plt.title("BSELoss vs Epoch - BreastMNIST")
# plt.legend(["Train Loss", "Validation Loss"])
# plt.grid()  
# plt.show()
# fig.savefig('cnn_loss.png')

# Plotting auc over all epochs
# fig, ax = plt.subplots()
# plt.plot(epoch_list, scores)
# plt.xlabel("Epoch")
# plt.ylabel("AUC Score")
# plt.title("AUC Score vs Epoch - BreastMNIST")
# plt.legend(["Validation"])
# plt.grid()  
# plt.show()
# fig.savefig('cnn_auc.png')


# Test set results based on epoch with highest accuracy (found during validation)
model.load_state_dict(torch.load('cnn_models/best_model.pth', weights_only=True))
tester = Tester(model, test_loader, criterion)
test_metrics = tester.test(threshold=0.6)
y_pred = test_metrics[1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate AUC
area = roc_auc_score(y_test, y_pred)

# Print metrics
print("Best Epoch: {}  |  Accuracy: {}, Precision: {}, Recall: {}, AUC: {}".format(best_epoch, accuracy, precision, recall, area))

# # Create confusion matrix for test set
# con_mat = confusion_matrix(y_test, y_pred)
# con_mat = pd.DataFrame(con_mat, columns=['Malignant (Predict)', 'Benign (Predict)'], 
#                                 index=['Malignant (True)', 'Benign (True)'])

# fig, ax = plt.subplots()
# sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
# plt.show()
# fig.savefig('cnn_confusion_matrix.png')
# plt.clf()