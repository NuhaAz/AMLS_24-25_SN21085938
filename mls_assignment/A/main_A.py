import numpy as np


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

X = np.vstack((data["train_images"], np.vstack((data["val_images"], data["test_images"]))))
Y = np.vstack((data["train_labels"], np.vstack((data["val_labels"], data["test_labels"]))))

