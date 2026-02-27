# Batch Gradient Descent for Multiple Linear Regression

This repository contains a custom Python implementation of **Batch Gradient Descent** built from scratch. It demonstrates how to train a Multiple Linear Regression model iteratively by calculating gradients and updating weights, rather than relying on a closed-form mathematical solution.

## 📄 Project Overview

The goal of this project is to understand the inner workings of gradient-based optimization. The notebook builds a custom `GDRegressor` class and compares its performance against Scikit-Learn's standard `LinearRegression` (which uses Ordinary Least Squares).

**Key Features:**
* Custom initialization of weights (coefficients) and bias (intercept).
* Vectorized implementation of the hypothesis function for efficiency.
* Calculation of gradients across the entire training batch simultaneously.
* Iterative updates using a configurable learning rate and epoch count.

## 📊 Dataset

The model is trained and evaluated on the standard **Diabetes Dataset** provided by Scikit-Learn:
* **Samples:** 442
* **Features:** 10 (age, sex, bmi, bp, etc., all mean-centered and scaled)
* **Target:** Quantitative measure of disease progression one year after baseline.

## 🧮 Mathematical Approach

Gradient Descent minimizes the Mean Squared Error (MSE) cost function by iteratively taking steps in the opposite direction of the gradient.

### 1. Hypothesis (Prediction)
$$\hat{y} = X \cdot w + b$$
Where $X$ is the feature matrix, $w$ are the coefficients, and $b$ is the intercept.

### 2. Gradients
To update the parameters, we calculate the partial derivatives of the cost function with respect to the intercept and the coefficients:

**Intercept Gradient:**
$$\frac{\partial L}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)$$

**Coefficients Gradient:**
$$\frac{\partial L}{\partial w} = -\frac{2}{N} X^T \cdot (y - \hat{y})$$
*(Where $N$ is the number of samples).*

### 3. Update Rule
$$b = b - \alpha \frac{\partial L}{\partial b}$$
$$w = w - \alpha \frac{\partial L}{\partial w}$$
*(Where $\alpha$ is the learning rate).*

## 🛠️ Implementation Details

The mathematical logic is translated into vectorized Numpy operations within the `GDRegressor` class:

```python
class GDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        # Initialize intercept and coefficients
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # Calculate predictions
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            
            # Calculate gradients
            intercept_der = -2 * np.mean(y_train - y_hat)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            
            # Update parameters
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            self.coef_ = self.coef_ - (self.lr * coef_der)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
