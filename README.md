🩺 Diabetes Prediction using Machine Learning
📌 Project Overview

This project focuses on predicting whether a patient has diabetes using machine learning classification algorithms.

The workflow includes:

✅ Data Collection

✅ Data Cleaning

✅ Data Wrangling

✅ Feature Engineering

✅ Model Building

✅ Model Evaluation & Comparison

The goal is to compare multiple machine learning models and determine which performs best for diabetes prediction.

📂 Dataset

The dataset contains medical diagnostic measurements used to predict diabetes.

🔢 Features:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

🎯 Target Variable:

Outcome

0 → Non-Diabetic

1 → Diabetic

🛠️ Data Preprocessing
🔹 Data Cleaning

Handled missing and zero values

Checked for duplicates

Verified data types

🔹 Data Wrangling

Feature selection

Data transformation

Train-Test split

Cross-validation applied

🤖 Models Implemented

The following machine learning algorithms were used:

Logistic Regression

Implemented with linear features

Polynomial feature transformation applied

Lasso (L1 Regularization) used for feature selection

Decision Tree Classifier

Random Forest Classifier

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

📊 Model Performance
Algorithm	Train Accuracy	Test Accuracy	Cross Validation
Logistic Regression	0.680	0.725	0.694
Decision Tree	1.000	0.550	0.580
Random Forest	1.000	0.680	0.670
Support Vector Classifier	0.680	0.725	0.694
K-Nearest Neighbors	0.733	0.600	0.620
📈 Key Observations

Decision Tree and Random Forest show overfitting (Train accuracy = 1.000).

Logistic Regression and SVC achieved the best test accuracy (72.5%).

Cross-validation results confirm that Logistic Regression and SVC are the most stable models.

Lasso regularization helped in reducing overfitting and selecting important features.

🏆 Best Performing Model

Based on test accuracy and cross-validation score:

👉 Logistic Regression and Support Vector Classifier performed best (72.5% test accuracy).

🧠 Technologies Used

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

📁 Project Structure
├── Diabetes_prediction.csv
├── diabetes_model.ipynb
├── README.md
└── requirements.txt
