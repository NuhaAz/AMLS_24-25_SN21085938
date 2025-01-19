import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader

# BloodMNIST dataset info
# Data          | Shape
#--------------------------------
# train_images  | (11959x3x28x28)
# train_labels  | (11959x3x1)
# val_images    | (1712x3x28x28)
# val_labels    | (1712x3x1)
# test_images   | (3421x3x28x28)
# test_labels   | (3421x3x1)
#--------------------------------
# Images = RGB , therefore 3 channels
# Basophil: 0, Eosinophil: 1, Erythroblast: 2,
# Immature Granulocytes: 3, Lymphocyte: 4,
# Monocyte: 5, Neutrophil: 6, Platelet: 7
#--------------------------------

def loaddata(file_loc):
    # Load data
    data = np.load(file_loc)


    # Convert 2D array (image) into 1D array of pixel values for each data split
    x_train = np.transpose(data["train_images"], (0, 3, 1, 2))
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], -1)
    y_train = np.array(data["train_labels"]).ravel()
    
    x_val = np.transpose(data["val_images"], (0, 3, 1, 2))
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], -1)
    y_val = np.array(data["val_labels"]).ravel()
    
    x_test = np.transpose(data["test_images"], (0, 3, 1, 2))
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], -1)
    y_test = np.array(data["test_labels"]).ravel()
    
    # Normalization using MinMaxScaler
    scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    x_train = np.array([scaler.fit_transform(x_train[i]) for i in range(x_train.shape[0])])
    x_val = np.array([scaler.fit_transform(x_val[i]) for i in range(x_val.shape[0])])
    x_test = np.array([scaler.fit_transform(x_test[i]) for i in range(x_test.shape[0])])
    
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
        data = torch.reshape(data, [length, 3, 28,28])
    elif setting == "y":
        data = torch.reshape(data, [length, 3])
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

def flatten_images(images):
    # Flatten the images to 2D array (num_samples, num_features)
    num_samples = images.shape[0]
    num_features = images.shape[1] * images.shape[2]
    return images.reshape(num_samples, num_features)

