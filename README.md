Repository for Applied Machine Learning System (ELEC0134) 24/25 Report.

**Summary:**
_BreastMNIST and BloodMNIST are medical imaging da-tasets used for machine learning (ML) classification. Three models were implemented due to their adaptability and ease to implement on any computer. The models implement-ed were: K-Nearest Neighbours (KNN), Support Vector Machines (SVM) and Convolutional Neural Networks (CNN)._
_Task A focused on the binary classification prob-lem associated with BreastMNIST. KNNs outperformed the other models with an accuracy of 81.41% and an AUC score of 76.00%. Task B focused on multi-class classifica-tion of BloodMNIST. Linear SVMs performed the best, achieving an accuracy score of 72.11%._

**Files:**

- main.py runs both Task A and Task B models sequentially
- Folders A and B contain a main python file (main_A.py or main_B.py), models.py and utils.py
- main_A/main_B.py executes all the models described in the report for the respective tasks
- models.py contains functions and classes for all the models and the classes required for CNN training, validation and training
- utils.py contains reusable functions relevant to the the execution of the code

**Python Libraries:**
PyTorch was used to create, train and evaluate the CNN. SKLearn was used for KNNs and SVMs and metric calculations.
Other relevant libraries for executing the code include: TQDM, NumPy, Pandas, Seaborn.
