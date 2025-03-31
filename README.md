# Siemens-forecast

## Project Overview

This project aims to develop a monthly sales forecasting model for fourteen product groups within a Business Unit of Siemens' Smart Infrastructure Division. The objective is to provide accurate sales predictions for the period May 2022 – February 2023, leveraging historical sales data and external macroeconomic indicators.

## Methodology

The project follows the CRISP-DM framework and includes:

1. Data Understanding & Preparation: Handling missing values, detecting outliers, and performing statistical tests
2. Feature Selection: Identifying key predictors for each product category
3. Modeling: Testing statistical (ARIMA, SARIMA, Prophet) and ML-based models, selecting the best based on RMSE & MAPE
4. Evaluation: Comparing model performance using RMSE and MAPE metrics
5. Deployment: Storing results and selecting the optimal forecasting model per product category

## Files & Structure

1. Folders
- data/ - data given to us for the project
- data/X_... - training, validation & test sets
- data/y_... - target datasets
- results/ - saved predictions

- extra data/ - data collected from the internet and transformed as necessary

2. Notebooks
01_EDA - Exploratory Data Analysis & Preprocessing
02_FS_&_modelling - Feature Selection and Modelling Phase
... (complete later)

3. Python Files
statistical_tests - functions for statistical tests
utils - useful dictionaries across notebooks
functions - mainly plots and other necessary functions
