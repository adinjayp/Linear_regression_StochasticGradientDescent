# Linear Regression Implementation Using True Stochastic Gradient Descent and Mini-Batch Stochastic Gradient Descent

This repository contains Python code for implementing linear regression from scratch and validating it against the `sklearn` library. Linear regression is a fundamental technique in statistical modeling for predicting a continuous variable based on one or more predictor variables.

## About the Code

The code consists of a class `LinearRegression` which implements linear regression using stochastic gradient descent, with an option for regularization and mini-batch optimization. It includes the following functionalities:

- Importing necessary libraries.
- Defining the `LinearRegression` class with methods for:
  - Normalizing training and testing data.
  - Stochastic gradient descent algorithm for model training.
  - Predicting output.
  - Calculating the cost function value.
  - Plotting the cost function value over iterations.
  - Splitting data into training and testing sets.
  - Calculating Root Mean Square Error (RMSE) and Sum of Square Error (SSE).
- Fetching the California Housing dataset.
- Creating instances of the `LinearRegression` class with and without regularization, and with different batch sizes.
- Training the models and printing results.

## Usage

You can use the provided code as follows:

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing

# Definition of LinearRegression class and methods...

# Performing True Stochastic Gradient Descent without Regularisation
cal_housing = fetch_california_housing()
lr = LinearRegression(cal_housing.data, cal_housing.target, learning_rate=0.0001, tolerance=1e-5, max_iterations=1000, regularization=False, lamda=0.05, batch_size=1)
lr.fit()

# Performing True Stochastic Gradient Descent with Regularisation
cal_housing = fetch_california_housing()
lr_reg = LinearRegression(cal_housing.data, cal_housing.target, learning_rate=0.0001, tolerance=1e-5, max_iterations=1000, regularization=True, lamda=0.001, batch_size=1)
lr_reg.fit()

# Performing Mini-Batch Stochastic Gradient Descent without Regularisation
cal_housing = fetch_california_housing()
lr_mini_batch = LinearRegression(cal_housing.data, cal_housing.target, learning_rate=0.0001, tolerance=1e-5, max_iterations=1000, regularization=False, lamda=0.05, batch_size=10)
lr_mini_batch.fit()

# Performing Mini-Batch Stochastic Gradient Descent with Regularisation
cal_housing = fetch_california_housing()
lr_mini_batch_reg = LinearRegression(cal_housing.data, cal_housing.target, learning_rate=0.0001, tolerance=1e-5, max_iterations=1000, regularization=True, lamda=0.001, batch_size=10)
lr_mini_batch_reg.fit()
