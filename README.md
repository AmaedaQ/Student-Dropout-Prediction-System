
```markdown
# Student Dropout Prediction System

## Overview

The **Student Dropout Prediction System** aims to predict the likelihood of student dropouts based on various factors such as demographics, academic performance, and financial status. This system uses multiple machine learning algorithms to classify students into categories: **Dropout**, **Graduated**, or **Currently Enrolled**. By identifying at-risk students, the system helps educational institutions take timely actions to improve student retention.

## Features

- **Machine Learning Models:** KNN, Logistic Regression, Decision Trees, Random Forest, SVM, Naive Bayes
- **Target Variable:** Dropout, Graduated, Currently Enrolled
- **Data:** Student demographics, academic performance, financial status, and more
- **Preprocessing:** Data cleaning, feature engineering, standardization, and PCA
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Dataset

The dataset consists of 4,424 students and 35 features. Key features include:
- **Demographic Information:** Age, Gender, Nationality
- **Academic Information:** Course performance, Curricular units
- **Financial Information:** Tuition fees, Scholarship status

### Target Variable Distribution:
- **Graduated:** 49%
- **Dropout:** 32.1%
- **Currently Enrolled:** 17%

## Models Used

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Support Vector Classifier (SVC)**
4. **Random Forest Classifier**
5. **K-Nearest Neighbors (KNN)**

## Installation

### Prerequisites

- Python 3.x
- Required libraries (listed in `requirements.txt`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AmaedaQ/Student-Dropout-Prediction.git
   cd Student-Dropout-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the system in your browser at:
   ```
   http://127.0.0.1:5000
   ```

## Model Evaluation

Here’s a brief evaluation of the models used:

- **Logistic Regression:** Achieved an accuracy of 81.0% on the test set.
- **Decision Tree Classifier:** Achieved an accuracy of 83.2%.
- **Random Forest Classifier:** Achieved an accuracy of 82.8% on the test set.
- **KNN:** Slightly lower accuracy at 79.5%, but good performance overall.

## Key Insights

- **Age & Gender:** Older male students had the highest dropout rates.
- **Financial Status:** Students with unpaid tuition or no scholarships had a significantly higher dropout probability.
  
## Future Work

- Hyperparameter tuning for better model performance.
- Integrating more features like academic performance metrics and social support systems.
- Enhanced analysis of gender-specific dropout trends for targeted interventions.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Scikit-learn, Flask, Pandas, NumPy, Matplotlib
- **Machine Learning Models:** Logistic Regression, Decision Trees, Random Forest, SVM, KNN
- **Data Preprocessing:** StandardScaler, PCA
- **Web Framework:** Flask (for the frontend)
