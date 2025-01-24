from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from torch.autograd import Variable


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
    params.setdefault('degree', 1)
    params.setdefault('kernel', 'linear')
    
    model = svm.SVC(C=params['C'], degree=params['degree'], kernel=params['kernel'])
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    return y_pred

class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super(CNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128 * 2*2,256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(128,1),
            nn.Sigmoid()
            )
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
                else:
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        exit()
        x = self.conv3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Train, Validation and Test Classes for CNN
class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        all_train_labels = []
        all_train_preds = []
        
        for i, (x_train, y_train) in enumerate(self.train_loader):
            x_train = Variable(x_train)
            y_train = Variable(y_train)
            self.optimizer.zero_grad()
            
            # Probability Prediction
            y_pred = self.model(x_train)
            
            loss = self.criterion(y_pred, y_train)
            train_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            # Collect all labels and predictions for precision and AUC score
            all_train_labels.extend(y_train.detach().numpy())
            all_train_preds.extend(y_pred.detach().numpy())
        
        avg_train_loss = train_loss / len(self.train_loader)
        train_accuracy = accuracy_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(float))
        train_precision = precision_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(float))
        train_auc = roc_auc_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(float))
        train_recall = recall_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(float))
        
        if self.scheduler:
            self.scheduler.step()
            
        # print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(self.train_loader)}, Train Accuracy: {train_accuracy} Train Precision: {train_precision}, Train Recall: {train_recall} Train AUC: {train_auc}")
        
        return avg_train_loss, train_accuracy, train_precision, train_auc, all_train_labels, all_train_preds
    
class Validator:
    def __init__(self, model, val_loader, criterion):
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion
        
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        all_val_labels = []
        all_val_preds = []
        
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(self.val_loader):
                x_val = Variable(x_val)
                y_val = Variable(y_val)
                
                # Probability Prediction
                y_pred = self.model(x_val)
                
                loss = self.criterion(y_pred, y_val)
                val_loss += loss.item()
                
                # Collect all labels and predictions for precision and F1-score
                all_val_labels.extend(y_val.detach().numpy())
                all_val_preds.extend(y_pred.detach().numpy())
                
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = accuracy_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(float))
        val_precision = precision_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(float))
        val_auc = roc_auc_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(float))
        val_recall = recall_score(all_val_labels,(np.array(all_val_preds) > 0.5).astype(float))
        
        # print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(self.val_loader)}, Val Accuracy: {val_accuracy}, Val Precision: {val_precision}, Val Recall: {val_recall} Val AUC: {val_auc}")
        
        return avg_val_loss, val_accuracy, val_precision, val_auc, all_val_labels, all_val_preds
    
class Tester:
    def __init__(self, model, test_loader, criterion):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        
    def test(self, threshold=0.5):
        self.model.eval()
        test_loss = 0
        all_test_labels = []
        all_test_preds = []
        class_0 = 0
        class_1 = 0
        
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(self.test_loader):
                x_test = Variable(x_test)
                y_test = Variable(y_test)
                
                y_pred = self.model(x_test)
                loss = self.criterion(y_pred, y_test)
                test_loss += loss.item()

                # Collect all labels and predictions for precision and F1-score
                all_test_labels.extend(y_test.detach().numpy())
                all_test_preds.extend(y_pred.detach().numpy())
                
        avg_test_loss = test_loss / len(self.test_loader)
        all_y_pred = all_test_preds
        all_y_pred_binary = (np.array(all_test_preds) > threshold).astype(float)
        
        return avg_test_loss, all_y_pred_binary, all_y_pred