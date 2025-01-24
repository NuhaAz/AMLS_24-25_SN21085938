import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
import torch.optim as optim
from tqdm import tqdm

# Import ML models
from models import *
from utils import *

print("\n")
print("{:-^50}".format(" Task B "))

# Load dataset
print("Loading BloodMNIST dataset...")
# file_loc = "../datasets/bloodmnist.npz"
file_loc = "datasets/bloodmnist.npz"
x_train, y_train, x_val, y_val, x_test, y_test = loaddata(file_loc)

# KNN Classifier
print("{:-^40}".format(" KNN Classifier "))

# Flatten dataset for KNN Classification
x_train = flatten_images(x_train)
x_val = flatten_images(x_val)
x_test = flatten_images(x_test)

# Training and validation of KNN model for different K Values

K = [i for i in range(1,int(np.sqrt(len(x_train))),int(np.sqrt(len(x_train)))//50)]
scores = []
for k in tqdm(K):
    y_pred = KNNClassifier(x_train, y_train, x_val, k)
    scores.append(accuracy_score(y_val, y_pred))

# Plotting validation set accuracy scores for different K Values
# fig, ax = plt.subplots()
# plt.plot(K, scores)
# plt.xlabel("K Value")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs K Value - BloodMNIST")
# plt.grid()  
# plt.show()
# fig.savefig('knn_accuracy.png')

# Test set results based on k with highest accuracy (found during validation)
index = scores.index(max(scores))
best_k = K[index]
y_pred = KNNClassifier(x_train, y_train, x_test, best_k)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro',zero_division=1)

# Print metrics
print("Best K Value: {}  |  Accuracy: {}, Precision: {}, Recall: {}".format(best_k, accuracy, precision, recall))

# Create confusion matrix for test set
# con_mat = confusion_matrix(y_test, y_pred)
# con_mat = pd.DataFrame(con_mat)
# columns=['Basophil (Predict)', 'Eosinophil (Predict)', 'Erythroblast (Predict)', 'Immature Granulocytes (Predict)',
#         'Lymphocyte (Predict)', 'Monocyte (Predict)', 'Neutrophil (Predict)', 'Platelet (Predict)'], 
# index=['Basophil (True)', 'Eosinophil (True)', 'Erythroblast (True)', 'Immature Granulocytes (True)',
#         'Lymphocyte (True)', 'Monocyte (True)', 'Neutrophil (True)', 'Platelet (True)']

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=45)
# plt.show()
# fig.savefig('knn_confusion_matrix.png')
# plt.clf()

# Linear SVM Classifier   
print("{:-^40}".format(" Linear SVM "))

# Training and Validation of SVM model

# Flatten dataset for SVM Classification
x_train = flatten_images(x_train)
x_val = flatten_images(x_val)
x_test = flatten_images(x_test)

# Validating for various gamma values
C_vals = [1e-1, 5e-1, 1, 5, 1e2]
scores = []
for C in tqdm(C_vals):
    params = {'C': C}
    y_pred = SVMClassifier(x_train, y_train, x_val, params)
    accuracy = accuracy_score(y_val, y_pred)
    scores.append(accuracy)
        
# Plotting validation set accuracy scores for different gamma Values
# fig, ax = plt.subplots()
# plt.plot(C_vals, scores)
# plt.xlabel("C Value")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs C Value - BloodMNIST")
# plt.grid()  
# plt.show()
# fig.savefig('svmlin_accuracy.png')

# Test set results based on gamma with highest accuracy (found during validation)
index = scores.index(max(scores))
param = {'C': C_vals[index]}
y_pred = SVMClassifier(x_train, y_train, x_test, param)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro',zero_division=1)

# Print metrics
print("Best Params: {}  |  Accuracy: {}, Precision: {}, Recall: {}".format(param, accuracy, precision, recall))

# Create confusion matrix
# con_mat = confusion_matrix(y_test, y_pred)
# con_mat = pd.DataFrame(con_mat)

# fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize parameter as needed
# sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=45)
# plt.show()
# fig.savefig('svmlin_confusion_matrix.png')
# plt.clf()
    
# CNN Classifier
print("{:-^40}".format(" CNN Classifier "))
    
# Hyperparameters
num_epochs = 50
batch_size = 32
lr = 0.001
lambda2 = 0.0001
dropout = 0.1

# Place data into train loaders for CNN inputs
print('Preparing data...')
train_loader = prepdata_cnn(x_train, y_train, batch_size)
val_loader = prepdata_cnn(x_val, y_val, batch_size)
test_loader = prepdata_cnn(x_test, y_test, batch_size)
torch.backends.cudnn.benchmark = True

# Create Model and training parameters
print("Creating model...")
model = CNN(dropout=dropout)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

trainer = Trainer(model, train_loader, criterion, optimizer, scheduler)
validator = Validator(model, val_loader, criterion)

# Settings for early stopping
patience = 5
best_accuracy = 0
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
    
    scores.append(val_metrics[1])
    train_loss_all.append(train_metrics[0])
    val_loss_all.append(val_metrics[0])
    
    if epoch+1 >= 15:
        # Early stopping
        accuracy = val_metrics[1]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
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
# plt.ylabel("Cross Entropy Loss")
# plt.title("Cross Entropy Loss vs Epoch - BloodMNIST")
# plt.legend(["Train Loss", "Validation Loss"])
# plt.grid()  
# plt.show()
# fig.savefig('cnn_loss.png')

# Plotting validation set accuracy scores for different epochs
# fig, ax = plt.subplots()
# plt.plot(epoch_list, scores)
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs Epoch - BloodMNIST")
# plt.grid()  
# plt.show()
# fig.savefig('cnn_accuracy.png')

# Test set results based on epoch with highest accuracy (found during validation)
model.load_state_dict(torch.load('cnn_models/best_model.pth'))
tester = Tester(model, test_loader, criterion)
test_metrics = tester.test(threshold=0.5)
y_pred = test_metrics[1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Print metrics
print("Best Epoch: {}  |  Accuracy: {}, Precision: {}, Recall: {}".format(best_epoch, accuracy, precision, recall))

# Create confusion matrix for test set
# con_mat = confusion_matrix(y_test, y_pred)
# con_mat = pd.DataFrame(con_mat)

# fig, ax = fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=45)
# plt.show()
# fig.savefig('cnn_confusion_matrix.png')
# plt.clf()