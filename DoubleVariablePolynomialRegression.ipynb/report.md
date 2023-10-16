# Comparative Analysis Report

### Polynomial Regression Model

Polynomial Regression is a type of regression analysis used in machine learning and statistics to model the relationship between a dependent variable (target) and one or more independent variables (features) as an nth-degree polynomial. It extends the simple linear regression model by allowing for non-linear relationships between the variables.

### Algorithms

The Polynomial Regression model is implemented using various algorithms, but the most common one is Gradient Descent. Here's an overview of the steps involved:

1. **Data Pre-Processing:** We first Loaded the data to a DataFrame and then Removed NAN/NULL Values and Normalized the data. We also Split the data into Test and Training Data sets
2. **Model Representation:** In Polynomial Regression, the relationship between the dependent variable (Y) and the independent variables (X1, X2) is represented as a polynomial equation of the form:

```
Y = w0 + w1*X1 + w2*X2 + w3*X1^2 + w4*X2^2 + w5*X1*X2... + wm*X1^n*X2^n + ε
```

2. **Cost Function:** To train the model, a cost function is defined, often Mean Squared Error (MSE). The goal is to minimize this cost function by adjusting the polynomial coefficients.
3. **Gradient Descent:** Gradient Descent is used to find the optimal values of the coefficients that minimize the cost function. It iteratively updates the coefficients in the direction of steepest descent of the cost function's gradient.
4. **Feature Engineering:** In Polynomial Regression, the independent variable (feature) X is often transformed to include polynomial terms of different degrees, such as X1^2, X1^2*X2^4, etc. This allows the model to capture non-linear patterns in the data.
5. **Model Training:** The model is trained using a dataset with known values of (X1, X2) and Y. During training, the coefficients are adjusted to minimize the cost function, resulting in the best-fitting polynomial curve.
6. **Model Evaluation:** After training, the model is evaluated using a separate test dataset to assess its predictive performance. Common evaluation metrics include Mean Squared Error, Root Mean Squared Error, and R-squared

### Implementation

In practice, the Polynomial Regression model is implemented as a Python class or function. Key steps include:

* Defining the polynomial degree (n) and initializing the coefficients (w0, w1, w2, ..., wm).
* Implementing the cost function (e.g., MSE) to measure the error between predicted and actual values.
* Implementing the Gradient Descent algorithm to update coefficients iteratively.
* Feature engineering by adding polynomial terms to the input features.
* Splitting the dataset into training and testing sets for model training and evaluation.
* Training the model by minimizing the cost function using Gradient Descent.
* Making predictions on new data points using the learned coefficients.
* Evaluating the model's performance using appropriate metrics.
* Using Regularization and Grid Search.

### Regularization Formula

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/Screenshot from 2023-10-16 16-59-23.png "a title")

where:

N = no of samples

lmbda, q = Regularization Hyperparameters

W = Weights

### Comparative Analysis

Each model is trained using Batch Gradient Descent, where the model's weights are updated iteratively to minimize the mean squared error. Training and testing errors are recorded for each epoch, and the best degree is determined based on these errors. The comparative analysis helps evaluate each model's performance and complexity, allowing for the selection of an appropriate polynomial degree for the given problem.

Let's perform a comparative analysis of the nine polynomial regression models we developed. We will discuss the results and provide an analysis for each degree of the polynomial:

Best models for each value of q:
q=0.5: Degree=7, Learning Rate=0.001, Lambda=0.5, Batch Size=20, Epoch=2571, Test Loss=139793.0962370393
q=1: Degree=7, Learning Rate=0.001, Lambda=0.0, Batch Size=20, Epoch=2571, Test Loss=139781.55365061367
q=2: Degree=7, Learning Rate=0.001, Lambda=1.0, Batch Size=20, Epoch=2571, Test Loss=125537.749596721
q=4: Degree=7, Learning Rate=0.001, Lambda=0.3, Batch Size=20, Epoch=2571, Test Loss=140085.22385513445

Best model for batch_size 1:
Degree=7, Learning Rate=0.001, Lambda=0.4, Batch Size=1, Epoch=0, Test Loss=363088.41025412834, q=4

Best model for batch_size 20:
Degree=7, Learning Rate=0.001, Lambda=1.0, Batch Size=20, Epoch=2571, Test Loss=125537.749596721, q=2

Best 4 models overall:
Rank 1: Degree=7, Learning Rate=0.001, Lambda=1.0, Batch Size=20, Epoch=2571, Test Loss=125537.749596721, q=2
Rank 2: Degree=7, Learning Rate=0.001, Lambda=0.5, Batch Size=20, Epoch=2571, Test Loss=139862.48767464896, q=2
Rank 3: Degree=7, Learning Rate=0.001, Lambda=0.1, Batch Size=20, Epoch=2571, Test Loss=139883.98162699374, q=2
Rank 4: Degree=7, Learning Rate=0.001, Lambda=0.2, Batch Size=20, Epoch=2571, Test Loss=139924.09367158657, q=2

### Overall Analysis

* The best degree of polynomial, based on minimum training error, is Degree 7. However, it has a significantly higher testing error, indicating overfitting.
* Lower-degree polynomials (Degree 0, Degree 1, Degree 2, Degree 3) are too simplistic and result in high errors.
* Higher-degree polynomials (Degree 8, Degree 9) tend to overfit the training data and do not generalize well to unseen data.

In summary, Degree 7 may be the most appropriate choices, depending on the desired trade-off between model complexity and generalization performance. Further tuning and regularization techniques can be applied to improve the models' generalization.

# Comparative Analysis of Best Degree Model and Regularized Models

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/all_subplots.png "a title")

# Surface Plots

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_2.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_3.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_4.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_5.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_6.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_7.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_8.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_degree_9.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_regularized_degree_7_q_2_lambda_0.1.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_regularized_degree_7_q_2_lambda_0.2.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_regularized_degree_7_q_2_lambda_0.5.png "a title")

![Alt text](/home/anirudh/Documents/GitHub/RegressionFromScratch/DoubleVariablePolynomialRegression.ipynb/surface_plot_regularized_degree_7_q_2_lambda_1.0.png "a title")
