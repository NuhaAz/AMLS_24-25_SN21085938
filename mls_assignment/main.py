from medmnist import BreastMNIST, BloodMNIST
import numpy as np
import subprocess

if __name__ == "__main__":
    print("{:-^50}".format(" Main "))
    
    # Download datasets
    # bloodmnist = BloodMNIST(split="train", download=True)
    # breastmnist = BreastMNIST(split="test", download=True)
    
    subprocess.run(["python", "A/main_A.py"])
    subprocess.run(["python", "B/main_B.py"])