---
title: Linear Regression
layout: post
---

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Linear Regression](#linear-regression)
  - [Ordinary Least Squares](#ordinary-least-squares)
  - [Regularization](#regularization)
    - [Bias-Variance Tradeoff](#bias-variance-tradeoff)
    - [Ridge Regression](#ridge-regression)
    - [Lasso Regression](#lasso-regression)

# Linear Regression

In classification problems, we try to classify input features into categories. But we may want to predict more diverse and continuous data. So instead of finding a function \\(f:\mathbb{R}^d\rightarrow\\{-1,1\\}\\), we want to model the function \\(f:\mathbb{R}^d\rightarrow\mathbb{R}\\).

## Ordinary Least Squares

Previously, we have been using the **Hinge Loss** as the objective for optimization because it takes the **product**, \\(y\hat{y}\\), as a measure of the similarity between the fitted value and the actual class and penalize it by substract it from 1 (100% similarity). However, the goal of regression is to predict a value rather than a class, so our loss function should somehow measure the **distance** between the predicted and actual value. For example, we can use the **Squared Loss**:

$$h(y,\hat{y}) = \frac{(y-\hat{y})^2}{2}$$

And for the entire training data, we measure the **Mean Squared Error** (MSE):

$$L(\mathbf{y}, \hat{\mathbf{y}})=\frac{1}{n}\sum_{i=1}^n{\frac{(y^{(i)}-\hat{y}^{(i)})^2}{2}}=\frac{1}{2n}(\mathbf{y}-\hat{\mathbf{y}})^\mathsf{T}(\mathbf{y}-\hat{\mathbf{y}})$$

So a common method for solving the linear regression problem is **Ordinary Least Squares** (OLS). We want to minimize the loss:

$$L(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}\sum_{i=1}^n{\frac{(y^{(i)}-\mathbf{w}\cdot\mathbf{x}^{(i)})^2}{2}}=\frac{1}{2n}(\mathbf{y}-\mathbf{X}\mathbf{w})^\mathsf{T}(\mathbf{y}-\mathbf{X}\mathbf{w})$$

We would like the gradient to shrink to 0:

$$
\begin{aligned}
\nabla_{\mathbf{w}}L(\mathbf{X}, \mathbf{y}, \mathbf{w})=&\;\frac{1}{2n}(2\mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}-2\mathbf{X}^\mathsf{T}\mathbf{y})=\frac{1}{n}(\mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}-\mathbf{X}^\mathsf{T}\mathbf{y})\\
\nabla_{\mathbf{w}^*}L(\mathbf{X}, \mathbf{y}, \mathbf{w})=&\;\frac{1}{n}(\mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}^*-\mathbf{X}^\mathsf{T}\mathbf{y})=\mathbb{0}\Rightarrow \mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}^*=\mathbf{X}^\mathsf{T}\mathbf{y}
\end{aligned}
$$

So, if \\(\mathbf{X}^\mathsf{T} \mathbf{X}\\) is a **Nonsingular** matrix, we can find a **closed-form solution** to the OLS regression problem (if not, there will be multiple solutions to the system \\(\mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}^*=\mathbf{X}^\mathsf{T}\mathbf{y}\\)):

$$\mathbf{w}^*=(\mathbf{X}^\mathsf{T} \mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}$$

But computing such matrix inverse operations can be expensive when the dimensions of \\(\mathbf{X}\\) is big. So as usual, we can use **gradient descent**.

*Pseudocode (SGD)*:
<pre>
k=0, w=0
<b>while</b> criterion not met:
    <b>for</b> i in 1...n:
        w = w - ηxi(w•xi-yi) // the Descent
        k++
</pre>

## Regularization
As we have seen in the closed-form solution, the matrix \\(\mathbf{X}^\mathsf{T}\mathbf{X}\\) may be singular. The underlying indication is that there are linearly correlated features. So some features may be redundant to our model. In fact, in machine learning and statistical analysis, we often prefer a simpler model. This is due to the tradeoff between bias and variance.

### Bias-Variance Tradeoff
We model the relationship between \\(y\\) and \\(\mathbf{x}\\) as \\(y=f(\mathbf{x})+\varepsilon\\), where \\(f\\) is the true underlying function and \\(\varepsilon\\) is the **irreducible error** with 0 mean and \\(\sigma^2\\) variance (\\(\Rightarrow\mathbb{E}[y]=f(\mathbf{x})\\)). 

**Bias** refers to the model's inability to correctly fit the data. Simple models tend to have higher bias. For example, a linear model applied to non-linear data would have high bias. When bias is high, the model will **underfit**. Statistically, it is defined as following:

$$\mathrm{Bias}(\hat{f}(\mathbf{x}))=\mathbb{E}[\hat{f}(\mathbf{x})]-f(\mathbf{x})$$

**Variance** refers to the model's sensitivity to noise in data. Complex models often have higher variance than simpler ones. For instance, a high degree polynomial model may have high variance because a slight shift in the input would result in a great change in the fitted value. When variance is high, the model would **overfit**. Statistically, it is defined as following:

$$\mathrm{Var}(\hat{f}(\mathbf{x}))=\mathbb{E}[(\hat{f}(\mathbf{x})-\mathbb{E}[\hat{f}(\mathbf{x})])^2]$$

If we calculate the expected squared error of the prediction, i.e. the **generalization error**:

$$
\begin{aligned}
\mathbb{E}[(y-\hat{f}(\mathbf{x}))^2]=&\;\mathbb{E}[(\underbrace{y-\mathbb{E}[y]}_{\varepsilon}+(\mathbb{E}[y]-\mathbb{E}[\hat{f}(\mathbf{x})])+(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x})))^2]\\
=&\;\mathbb{E}[\varepsilon^2]+\mathbb{E}[(\mathbb{E}[y]-\mathbb{E}[\hat{f}(\mathbf{x})])^2]+\mathbb{E}[(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x}))^2]\\
&+2\mathbb{E}[\varepsilon(\mathbb{E}[y]-\mathbb{E}[\hat{f}(\mathbf{x})])]+2\mathbb{E}[\varepsilon(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x}))]\\
&+2\mathbb{E}[(\mathbb{E}[y]-\mathbb{E}[\hat{f}(\mathbf{x})])(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x}))]\\
=&\;\sigma^2+(\underbrace{\mathbb{E}[y]}_{f(\mathbf{x})}-\mathbb{E}[\hat{f}(\mathbf{x})])^2+\mathrm{Var}(\hat{f}(\mathbf{x}))\\
&+2\underbrace{\mathbb{E}[\varepsilon]}_{0}(\mathbb{E}[y]-\mathbb{E}[\hat{f}(\mathbf{x})])+2\underbrace{\mathbb{E}[\varepsilon(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x}))]}_{\varepsilon, \hat{f}\text{ independent}}\\
&+2(\mathbb{E}[y]-\mathbb{E}[\hat{f}(\mathbf{x})])\underbrace{\mathbb{E}[(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x}))]}_{0}\\
=&\;\sigma^2+(f(\mathbf{x})-\mathbb{E}[\hat{f}(\mathbf{x})])^2+\mathrm{Var}(\hat{f}(\mathbf{x}))\\
&+2\mathbb{E}[\varepsilon]\mathbb{E}[(\mathbb{E}[\hat{f}(\mathbf{x})]-\hat{f}(\mathbf{x}))]\\
=&\;\sigma^2+\mathrm{Bias}(\hat{f}(\mathbf{x}))^2+\mathrm{Var}(\hat{f}(\mathbf{x}))
\end{aligned}
$$

Their relationship with the model complexity is shown as <a href="https://medium.com/@rsehrawat75/bias-variance-tradeoff-f0e3afb78879">following</a>:
<p align="center">
<img src="https://djsaunde.files.wordpress.com/2017/07/bias-variance-tradeoff.png" width=500>
</p>
Since the bias decreases while variance increases with model complexity, we need to find a middle ground so that the generalization error can be minimized.

For the OLS estimator, if we look at the bias and variance:

$$
\begin{aligned}
\mathrm{Bias}(\hat{\mathbf{w}})=&\;\mathbb{E}[\hat{\mathbf{w}}]-\mathbf{w}\\
=&\;\mathbb{E}[(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}]-\mathbf{w}\\
=&\;(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\underbrace{\mathbb{E}[\mathbf{y}]}_{f(\mathbf{x})}-\mathbf{w}\\
=&\;(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{X}\mathbf{w}-\mathbf{w}\\
=&\;\mathbf{w}-\mathbf{w}=\mathbf{0}\\
\mathrm{Var}(\hat{\mathbf{w}})=&\;\mathbb{E}[(\hat{\mathbf{w}}-\mathbb{E}[\hat{\mathbf{w}}])^2]\\
=&\;\mathbb{E}[((\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}-\mathbb{E}[(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}])^2]\\
=&\;(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbb{E}[\underbrace{(\mathbf{y}-\mathbb{E}[\mathbf{y}])}_{[ε, ..., ε]^\mathsf{T}}(\mathbf{y}-\mathbb{E}[\mathbf{y}])^\mathsf{T}]((\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T})^\mathsf{T}\\
=&\;(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathrm{Var}(\textbf{ε})\mathbf{X}((\mathbf{X}^\mathsf{T}\mathbf{X})^{-1})^\mathsf{T}\\
=&\;(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}(\sigma^2\mathbf{I})\mathbf{X}((\mathbf{X}^\mathsf{T}\mathbf{X})^\mathsf{T})^{-1}\\
=&\;\sigma^2(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{X}(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\\
=&\;\sigma^2(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\\
\mathbb{E}[(y-\hat{y})^2]=&\;\sigma^2((\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}+1)
\end{aligned}
$$

So the OLS estimator is unbiased. But that does not mean that there is no tradeoff for OLS. Because in real life it is hard for the estimator to be directly calculated. Either the comptation is too expensive or there are correlated features. In practice, iterative approximation approached like SGD are used so there will be bias existing in the training.

To counter the potential of overfitting, **Regularization** is introduced. Not only do we want to minimize the loss, we would also like to limit the complexity of the model. So the new objective function of our regression problem looks like:

$$J(X,\mathbf{y},\mathbf{w})=L(\mathbf{X},\mathbf{y},\mathbf{x})+\lambda R(\mathbf{w})$$

where \\(\lambda\\) controls the extend to which we want to regularize the model.

### Ridge Regression
Their are a few ways to limit the value of our model parameters. One way is to use the L2 penalty (here we divide by \\(2n\\) for mathematical simplicity, it does not matter a lot in practice):

$$R(\mathbf{w})=\frac{\left\|\mathbf{w}\right\|^2}{2n}$$

This kind of regression with L2 regularization is called the **Ridge Regression**:

$$J(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{2n}(\mathbf{y}-\mathbf{X}\mathbf{w})^\mathsf{T}(\mathbf{y}-\mathbf{X}\mathbf{w})+\frac{\lambda}{2n}\left\|\mathbf{w}\right\|^2$$

Now the gradient of the objective function is:

$$
\begin{aligned}
\nabla_{\mathbf{w}}J(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}(\mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}-\mathbf{X}^\mathsf{T}\mathbf{y})+\frac{\lambda}{n}\mathbf{w}
\end{aligned}
$$

Similar to OLS, we can obtain a closed-form solution to the optimal model:

$$
\begin{aligned}
\nabla_{\mathbf{w}^*}J(\mathbf{X}, \mathbf{y}, \mathbf{w})&=\frac{1}{n}(\mathbf{X}^\mathsf{T}\mathbf{X}\mathbf{w}^*-\mathbf{X}^\mathsf{T}\mathbf{y})+\frac{\lambda}{n}\mathbf{w}^*=0\\
\mathbf{w}^*&=(\mathbf{X}^\mathsf{T}\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}
\end{aligned}
$$

Note that this time \\(\mathbf{X}^\mathsf{T}\mathbf{X}+\lambda\mathbf{I}\\) will always be nonsingular as long as \\(\lambda>0\\). \\(\mathbf{X}^\mathsf{T}\mathbf{X}\\) is always **positive semidefinite**, and a matrix is invertible iff it is **postive definite**. Since the sum of a positive definite matrix and a positive semidefinite matrix is positve definite, \\(\mathbf{X}^\mathsf{T}\mathbf{X}+\lambda\mathbf{I}\\) will always be positive definite and thus nonsingular. Therefore Ridge Regression is usually chosen when there are hign collinearity in the data.

### Lasso Regression
Although Ridge Regression solve the problem of collinearity, it will still produce a rather complex model because when the value of \\(w^{(i)}\\) is small, it will get less penalized by the L2 term. If we want the model to eliminate some useless parameters, we would need "harsher" penalize. So instead of using the squared weight, we use the absolute value, i.e. the L1 penalty:

$$R(\mathbf{w})=\frac{|\mathbf{w}|}{n}$$

$$J(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{2n}(\mathbf{y}-\mathbf{X}\mathbf{w})^\mathsf{T}(\mathbf{y}-\mathbf{X}\mathbf{w})+\frac{\lambda}{n}|\mathbf{w}|$$

Now the gradient would be:

$$
\begin{aligned}
\nabla_{\mathbf{w}}J(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}(\mathbf{X}^\mathsf{T} \mathbf{X}\mathbf{w}-\mathbf{X}^\mathsf{T}\mathbf{y})+\frac{\lambda}{n}\mathrm{sgn}(\mathbf{w})
\end{aligned}
$$