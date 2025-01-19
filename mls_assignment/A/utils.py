import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader

# BloodMNIST dataset info
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

def loaddata(file_loc):
    # Load data
    data = np.load(file_loc)

    # Convert 2D array (image) into 1D array of pixel values for each data split
    x_train = np.array([np.hstack(x) for x in data["train_images"]])
    y_train = np.array(data["train_labels"]).ravel()
    x_val = np.array([np.hstack(x) for x in data["val_images"]])
    y_val = np.array(data["val_labels"]).ravel()
    x_test = np.array([np.hstack(x) for x in data["test_images"]])
    y_test = np.array(data["test_labels"]).ravel()
    
    # Normalization using MinMaxScaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_test = scaler.fit_transform(x_test)
    
    # Convert to tensor
    x_train = torch.from_numpy(x_train)
    x_train = x_train.to(torch.float32)
    y_train = torch.from_numpy(y_train)
    y_train = y_train.to(torch.float32)
    x_val = torch.from_numpy(x_val)
    x_val = x_val.to(torch.float32)
    y_val = torch.from_numpy(y_val)
    y_val = y_val.to(torch.float32)
    x_test = torch.from_numpy(x_test)
    x_test = x_test.to(torch.float32)
    y_test = torch.from_numpy(y_test)
    y_test = y_test.to(torch.float32)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# Reshape data for CNN inputs
def reshape(data, setting):
    length = data.shape[0]
    if setting == "x":
        data = torch.reshape(data, [length, 1, 28,28])
    elif setting == "y":
        data = torch.reshape(data, [length, 1])
    return data

# Prepare CNN dataset
def prepdata_cnn(x, y, batch_size):
    # Reshape dataset for CNN: images (x) to 4D [len, 1, 28, 28], labels (y) to 2D [len, 1]
    x = reshape(x, "x")
    y = reshape(y, "y")
    
    # Place data into train loaders for CNN inputs
    dataset = [(x[i], y[i])for i in range(x.size(0))]
    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    return loader
