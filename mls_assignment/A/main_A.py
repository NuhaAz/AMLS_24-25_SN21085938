import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Load data
file_loc = "../datasets/breastmnist.npz"
data = np.load(file_loc)
# print(data["test_labels"].shape)
# Data          | Shape
#----------------------------
# train_images  | (546x28x28)
# train_labels  | (546x1)
# val_images    | (78x28x28)
# val_labels    | (78x1)
# test_images   | (156x28x28)
# test_labels   | (156x1)

# Convert 2D array (image) into 1D array of pixel values
X = np.vstack((np.array([np.hstack(x) for x in data["train_images"]]),
                np.vstack((np.array([np.hstack(x) for x in data["val_images"]]),
                          np.array([np.hstack(x) for x in data["test_images"]])))
                ))
Y = np.vstack((data["train_labels"], np.vstack((data["val_labels"], data["test_labels"]))))
Y = np.ravel(Y)

# Shuffle and split the dataset
X, Y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5, random_state=0)
# print('train set: {}  | test set: {}'.format(round(((len(y_train)*1.0)/len(X)),3),
#                                                        round((len(y_test)*1.0)/len(X),3)))

# KNNClassifier function
def KNNClassifier(x_train, y_train, x_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train) # Fit KNN model

    y_pred = neigh.predict(x_test)
    return y_pred


# Training and validation of KNN model
K = [i for i in range(1,int(np.sqrt(len(x_train))))]
scores = []
for k in K:
    y_pred= KNNClassifier(x_train, y_train, x_test,k)
    scores.append(accuracy_score(y_test,y_pred))

fig, ax = plt.subplots()
plt.plot(K, scores)  
plt.show()
fig.savefig('accuracy.png')
plt.clf()

# Testing model on k with highest accuracy
index = scores.index(max(scores))
k = K[index]
y_pred= KNNClassifier(x_train, y_train, x_test,k)

con_mat = confusion_matrix(y_test, y_pred)

con_mat = pd.DataFrame(con_mat, columns=['Cancer (True)', 'Malignant (True)'], 
                                 index=['Cancer (Predict)', 'Malignant (Predict)'])

fig, ax = plt.subplots()
sns.heatmap(con_mat, annot=True, fmt='d', cmap='PuBu')
plt.show()
fig.savefig('confusion_matrix.png')
plt.clf()