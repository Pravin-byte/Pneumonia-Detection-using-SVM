# Pneumonia Detection using SVM and Ensemble Learning

This project focuses on detecting **pneumonia** from chest X-ray images using classical Machine Learning techniques. The model uses **Support Vector Machines (SVM)** with **PCA for dimensionality reduction**, and explores ensemble methods like **Bagging**, **Voting Classifier**, and **AdaBoost** to enhance accuracy.

---

##  Features

-  Image classification: Normal vs Pneumonia
-  Dimensionality reduction using PCA
-  SVM model tuning with GridSearchCV
-  Ensemble models for performance boost
-  Achieves high accuracy (~97.22%) on test set

---

## Tech Stack

- Python
- Scikit-learn
- NumPy, Matplotlib, Pillow
- Jupyter Notebook

---

## Folder Structure

Pneumonia-Detection-using-SVM/
├── main.ipynb                
├── Optimized_SVM_Pneumonia_Model.joblib 
├── README.md                  
├── requirements.txt           
└── .gitignore 

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection-svm.git
   cd pneumonia-detection-svm

2. Install dependencies:

   pip install -r requirements.txt

3.Run the notebook: Open main.ipynb in Jupyter Notebook or Jupyter Lab and run all cells.


## Results

Model	  			Accuracy

SVM (PCA + GridSearch)		97.22%
Bagging (SVM)			97.13%
Voting Classifier		96.36%
AdaBoost			95.21%


## Dataset

Dataset used:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
You must download it separately due to size.


##streamlit app
https://pneumonia-detection-using-svm-ketnxdvxymzppufzecpjha.streamlit.app/


