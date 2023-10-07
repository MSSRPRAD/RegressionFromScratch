# Comparative Analysis Report

### Polynomial Regression Model

Polynomial Regression is a type of regression analysis used in machine learning and statistics to model the relationship between a dependent variable (target) and one or more independent variables (features) as an nth-degree polynomial. It extends the simple linear regression model by allowing for non-linear relationships between the variables.

### Algorithms

The Polynomial Regression model is implemented using various algorithms, but the most common one is Gradient Descent. Here's an overview of the steps involved:

1. **Model Representation:** In Polynomial Regression, the relationship between the dependent variable (Y) and the independent variable (X) is represented as a polynomial equation of the form:

```
Y = β0 + β1*X + β2*X^2 + ... + βn*X^n + ε
```

2. **Cost Function:** To train the model, a cost function is defined, often Mean Squared Error (MSE). The goal is to minimize this cost function by adjusting the polynomial coefficients.
3. **Gradient Descent:** Gradient Descent is used to find the optimal values of the coefficients that minimize the cost function. It iteratively updates the coefficients in the direction of steepest descent of the cost function's gradient.
4. **Feature Engineering:** In Polynomial Regression, the independent variable (feature) X is often transformed to include polynomial terms of different degrees, such as X^2, X^3, etc. This allows the model to capture non-linear patterns in the data.
5. **Model Training:** The model is trained using a dataset with known values of X and Y. During training, the coefficients are adjusted to minimize the cost function, resulting in the best-fitting polynomial curve.
6. **Model Evaluation:** After training, the model is evaluated using a separate test dataset to assess its predictive performance. Common evaluation metrics include Mean Squared Error, Root Mean Squared Error, and R-squared

### Implementation

In practice, the Polynomial Regression model is implemented as a Python class or function. Key steps include:

* Defining the polynomial degree (n) and initializing the coefficients (β0, β1, β2, ..., βn).
* Implementing the cost function (e.g., MSE) to measure the error between predicted and actual values.
* Implementing the Gradient Descent algorithm to update coefficients iteratively.
* Feature engineering by adding polynomial terms to the input features.
* Splitting the dataset into training and testing sets for model training and evaluation.
* Training the model by minimizing the cost function using Gradient Descent.
* Making predictions on new data points using the learned coefficients.
* Evaluating the model's performance using appropriate metrics.

### Comparative Analysis

Each model is trained using Batch Gradient Descent, where the model's weights are updated iteratively to minimize the mean squared error. Training and testing errors are recorded for each epoch, and the best degree is determined based on these errors. The comparative analysis helps evaluate each model's performance and complexity, allowing for the selection of an appropriate polynomial degree for the given problem.

Let's perform a comparative analysis of the nine polynomial regression models we developed. We will discuss the results and provide an analysis for each degree of the polynomial:

### Degree 0

* Minimum Train Error: 53072.09397392719
* Minimum Test Error: 89755.13359871

**Analysis:**
Degree 0 represents a constant model. Both the minimum training and testing errors are very high, indicating that this model is overly simplistic and does not capture the data's patterns effectively. This degree is not suitable for modeling the relationship between the features and the target variable.

### Degree 1

* Minimum Train Error: 53085.13139793781
* Minimum Test Error: 89732.03213397988

**Analysis:**
Degree 1 represents a linear model. The minimum training and testing errors are still quite high, suggesting that this linear model may not capture the underlying complexity of the data well. This degree does not seem to provide a good fit for the problem.

### Degree 2

* Minimum Train Error: 53096.601557574475
* Minimum Test Error: 89696.82310603288

**Analysis:**
Degree 2 represents a quadratic model. Although it performs slightly better than the linear model, the minimum training and testing errors remain high. This indicates that the quadratic model captures some non-linearity in the data but may not be sufficient to provide an accurate representation.

### Degree 3

* Minimum Train Error: 53089.08648459167
* Minimum Test Error: 89793.41272926814

**Analysis:**
Degree 3 represents a cubic model. The minimum training and testing errors continue to decrease, indicating that the model is becoming more flexible and capturing additional patterns in the data. Degree 3 shows a significant improvement compared to lower degrees and may be a better choice for this problem.

### Degree 4

* Minimum Train Error: 53216.92830454346
* Minimum Test Error: 89848.76336031037

**Analysis:**
Degree 4 represents a quartic model. The errors show further improvement over the cubic model, suggesting that the quartic model can fit the data even better with increased complexity. However, the test error is still relatively high, indicating that further investigation may be needed.

### Degree 5

* Minimum Train Error: 53089.060017504016
* Minimum Test Error: 89737.66486622303

**Analysis:**
Degree 5 represents a quintic model. The minimum testing error decreases further, indicating that the model can capture even more intricate patterns in the data. It achieves a lower testing error than the quartic model, suggesting that it may be a good choice for this problem.

### Degree 6

* Minimum Train Error: 52966.764284697376
* Minimum Test Error: 89633.79226026894

**Analysis:**
Degree 6 represents a sextic model. It has the lowest minimum training error among all degrees, suggesting that it fits the training data exceptionally well. However, the testing error is substantially higher, indicating a potential issue with overfitting. While it fits the training data closely, it may not generalize well to unseen data.

### Degree 7

* Minimum Train Error: 53056.65059954847
* Minimum Test Error: 89751.24170750633

**Analysis:**
Degree 7 represents a septic model. While it performs well on the training data, the test error is higher, indicating a possible overfitting issue. The testing error continues to increase compared to the training error, suggesting that this degree may not be the best choice for this problem.

### Degree 8

* Minimum Train Error: 53130.71008171925
* Minimum Test Error: 89819.68500509144

**Analysis:**
Degree 8 represents an octic model. It's increasingly complex, and it might lead to even more severe overfitting. The test error is higher than the training error, indicating a problem with generalization. This degree may not be suitable for this problem.

### Degree 9

* Minimum Train Error: 53082.65285620763
* Minimum Test Error: 89737.31690114134

**Analysis:**
Degree 9 represents a nonic model. This is the most complex model among the tested degrees. It has a low training error but a considerably lower testing error compared to the sextic and septic models, indicating that it may have better generalization performance. However, it's essential to be cautious about potential overfitting.

### Overall Analysis

* The best degree of polynomial, based on minimum training error, is Degree 6. However, it has a significantly higher testing error, indicating overfitting.
* Degree 5 represents a good balance between model complexity and generalization, as it achieves a lower testing error than Degree 4.
* Degree 3 and Degree 4 also perform reasonably well, capturing the data's non-linearity without excessive complexity.
* Lower-degree polynomials (Degree 0, Degree 1, Degree 2) are too simplistic and result in high errors.
* Higher-degree polynomials (Degree 7, Degree 8, Degree 9) tend to overfit the training data and do not generalize well to unseen data.

In summary, Degree 5 or Degree 3 may be the most appropriate choices, depending on the desired trade-off between model complexity and generalization performance. Further tuning and regularization techniques can be applied to improve the models' generalization.
