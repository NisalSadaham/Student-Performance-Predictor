# Student Performance Predictor

A machine learning project that predicts students' Performance Index using academic and lifestyle-related features.

## Project Overview

The goal of this project was to gain practical experience with the machine learning workflow, particularly multiple linear regression using Python and Scikit-learn.

I explored the dataset, performed exploratory data analysis, prepared the data, trained a regression model, and evaluated its performance using different regression metrics.

## Dataset

Dataset: [Student Performance – Multiple Linear Regression (Kaggle)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)

The dataset contains the following input features:

- **Hours Studied** – Number of hours spent studying.
- **Previous Scores** – Student's previous academic scores.
- **Extracurricular Activities** – Whether the student participates in extracurricular activities.
- **Sleep Hours** – Average hours of sleep.
- **Sample Question Papers Practiced** – Number of practice papers completed.

**Target variable:** `Performance Index`

## Project Workflow

### 1. Data Understanding
- Loaded and inspected the dataset using Pandas.
- Examined dataset structure, data types, and summary statistics.
- Checked for duplicate records.

### 2. Exploratory Data Analysis (EDA)
- Analysed the distribution of Performance Index.
- Created visualisations using Matplotlib.
- Examined relationships between input features and the target variable.
- Used correlation analysis and a correlation heatmap to understand feature relationships.

### 3. Data Preprocessing
- Prepared a cleaned dataset.
- Removed duplicate records.
- Converted categorical values in `Extracurricular Activities` into numerical values.

### 4. Model Development
- Implemented multiple linear regression using Scikit-learn.
- Split the dataset into training and testing sets (80/20).
- Trained the model using the training data.
- Generated predictions using the testing data.
- Manually verified a prediction using NumPy and the learned linear regression equation.

### 5. Model Evaluation

The model was evaluated using:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.6470 |
| Mean Squared Error (MSE) | 4.3059 |
| Root Mean Squared Error (RMSE) | 2.0751 |
| R² Score | 0.9884 |
| Training R² | 0.9887 |
| Testing R² | 0.9884 |

The model achieved an R² score of **0.9884** on the test set, indicating a strong fit to the test data.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn


## Key Learning Outcomes

Through this project, I gained practical experience with:

- Exploratory Data Analysis.
- Data cleaning and preprocessing.
- Correlation analysis.
- Multiple Linear Regression.
- Training and testing datasets.
- Regression evaluation metrics.
- Model coefficients and intercept interpretation.
- Making predictions using Scikit-learn.
- Understanding the mathematical calculations behind model predictions.

## Future Improvements

- Perform residual analysis.
- Investigate feature importance and multicollinearity.
- Experiment with other regression algorithms.
- Explore hyperparameter tuning and cross-validation.
- Build a simple interface to make predictions.

## License and Dataset Attribution

The dataset was obtained from Kaggle:

[Student Performance – Multiple Linear Regression](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)

According to the dataset description, the data is free to share and use.

This project is intended for educational purposes.

## Disclaimer

This project was developed for educational purposes as part of my journey in learning AI and Machine Learning.