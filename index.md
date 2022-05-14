---
---

{% include head.html %}
<link rel="stylesheet" href="/ml-replica/assets/css/index.css">

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Linear Clasification](#linear-clasification)
  - [Perceptron](#perceptron)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
    - [Loss Functions](#loss-functions)
    - [Gradient Descent](#gradient-descent)
    - [Stochastic Gradient Descent](#stochastic-gradient-descent-1)
  - [Support Vector Machine](#support-vector-machine)
    - [Maximum Margin Separator](#maximum-margin-separator)
    - [Hard-Margin SVM](#hard-margin-svm)
    - [Lagrange Duality](#lagrange-duality)
    - [Soft-Margin SVM](#soft-margin-svm)
    - [Feature Mapping](#feature-mapping)
    - [Kernel Trick](#kernel-trick)
    - [Sequential Minimal Optimization](#sequential-minimal-optimization)
- [Regression](#regression)
  - [Ordinary Least Squares](#ordinary-least-squares)
  - [Regularization](#regularization)
    - [Ridge Regression](#ridge-regression)
    - [Lasso Regression](#lasso-regression)

# Linear Clasification
Linear classifiers classifies the input features based on the decision hyperplane in the feature space.

## Perceptron
The perceptron algorithm is the building block of deep learning. It updates on one data point at each time and moves in the "right" direction based on that point. <br><br>
*Pseudocode* (w/o offset)
<pre>
k=0, w=0
<b>while</b> not all correctly classified <b>and</b> k < max step:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w•xi) <= 0: // misclassified
            w = w + yixi // the Update
            k++
</pre>
*Pseudocode* (w/ offset)
<pre>
k=0, w=0, b=0
<b>while</b> not all correctly classified <b>and</b> k < max step:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w.Xi+b) <= 0: // misclassified
            w = w + yiXi // the Update
            b = b + yi // offset update
            k++
</pre>
*Note*: we can convert the w/ offset version to w/o by transforming `X` to `[1, X]`, then the first resulting weight parameter would be the offset.

*Code*: [algorithm.py - perceptron()](https://github.com/xianglous/ml-replica/tree/main/mlreplica/utils/algorithm.py#L5)

## Stochastic Gradient Descent
Perceptron is nice and simple, but it has an important restriction: it only converges on linearly-separable data. <br>
To make it work for non-separable data, we need to change the way it approaches the best model. 

### Loss Functions
In machine learning, we often use a loss function to measure the fit of the current model to the training data. For the perceptron algorithm, we want our fitted value to have the same sign as the actual class, so we multiply them together and penalize those with negative products. In another word, the perceptron algorithm uses the following loss function:

$$L(X, \bar{y}, \bar{w})=\frac{1}{n}\sum_{i=1}^n{[y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})\leq 0]}$$

A problem with this loss function is that it does not measures the distance between the predicted value and actual class, so 0.1 and 1 will all be seen as a good classification while -0.1 and -1 will all be equally bad. <br>

So another loss function we can use instead is the **Hinge Loss**, for a fitted value \\(\hat{y}\\), the Hingle Loss is:

$$h(y,\hat{y})=\max(0, 1-y\hat{y})$$

For our linear model, the loss function is defined as:

$$L(X, \bar{y}, \bar{w})=\frac{1}{n}\sum_{i=1}^n{\max(0, 1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})))}$$

This loss will penalize any **imperfect** prediction.

### Gradient Descent
The loss function tells us about how **bad** the current model fits the data. Therefore, we need to know the direction in which moving the parameters will decrease the loss. In mathematics, we use the gradient to measure the "direction." For Hinge Loss, the gradient of a single data point is: 

$$\nabla_{\bar{w}}{h(\bar{x}^{(i)}, y^{(i)},\bar{w})}=
    \begin{cases}
    -y^{(i)}\bar{x}^{(i)}&\text{if }y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})<1\\
    \mathbf{0} & \text{otherwise}
    \end{cases}$$

And the gradient of the whole training data is:

$$\nabla_{\bar{w}}{L(X, \bar{y},\bar{w})}=\frac{1}{n}\sum_{i=1}^n{\nabla_{\bar{w}}{h(\bar{x}^{(i)}, y^{(i)},\bar{w})}}$$

By moving the the weights in the direction of the gradient, we will likely decrease the loss of our model. So the **Gradient Descent** is defined as:

$$\bar{w}^{(k+1)}=\bar{w}^{(k)}-\eta\nabla_{\bar{w}}{L(X, \bar{y}, \bar{w})}$$

*Pseudocode*
<pre>
k=0, w=0
<b>while</b> criterion not met:
    g = 0
    <b>for</b> i in 1...n:
        <b>if</b> yi(w•xi)<1:
            g = g - yixi/n
    w = w - η*g // the Descent
    k++
</pre>
η is the step size, or the learning rate.

### Stochastic Gradient Descent
The problem with gradient descent is that we need to compute the gradient of each data point in every iteration, which can be slow when the training data is huge. Alternatively, we can update based on a single data point in each iteration, and that is **Stochastic Gradient Descent**.<br>

*Pseudocode*
<pre>
k=0, w=0
<b>while</b> criterion not met:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w•xi)<1:
            w = w + η*yixi // the Descent
            k++
</pre>

*Code*: [algorithm.py - SGD() / GD()](https://github.com/xianglous/ml-replica/tree/main/mlreplica/utils/algorithm.py#L39)

## Support Vector Machine
As mentioned before, in linear classification problems we want to find a hyperplane that separates training data well. But there can be infinitely many hyperplanes that separate the data, we need to have additional measures to select the best ones. 

### Maximum Margin Separator
Consider the following example:

<p align="center">
<img src=" /ml-replica/assets/images/non_max_margin.png" width=300/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src=" /ml-replica/assets/images/max_margin.png" width=300/>
</p>

The solid lines on each figure represents a classification boundary that separates the two training classes. They both perfectly classify the training data, but which one is better? We would consider the right one better because it maximizes the "margins" between the separator and the training data.<br>
So while using the Hinge Loss may produce either of the above models, maximizing the margin will give us the better model. In another word, we want to maximize the distance from the closest point to the separating hyperplane:

<p align="center">
<img src=" /ml-replica/assets/images/max_margin_distance.png" width=300/>
</p>

If we look at the two margine lines, they are actually the decision lines \\(\bar{w}\cdot\bar{x}+b=1\\) and \\(\bar{w}\cdot\bar{x}+b=-1\\) beecause they are right on the border of being penalized. So we can calculate the margin as the distance between the positive margin line and the decision boundary:

$$d=\frac{|(1-b)-(-b)|}{\left\|\bar{w}\right\|}=\frac{1}{\left\|\bar{w}\right\|}$$

And we want our model to maximize the margin:

$$\displaystyle\max_{\bar{w}, b}{\frac{1}{\left\|\bar{w}\right\|}}$$

### Hard-Margin SVM
We can now formulate our problem as a constrained optimization. For computation purpose, we transform the maximization into a minimization problem:

$$\begin{aligned}
\displaystyle\min_{\bar{w}}\;\;&{\frac{ {\left\|\bar{w}\right\|}^2}{2}}\\
\text{ subject to }\;&y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1,\forall i\in\{1,...n\}
\end{aligned}$$

### Lagrange Duality
For a constrained optimization problem \\(\min_{\bar{\theta}}{f(\bar{\theta})}\\) subject to \\(\eta\\) constraints $h_i(\bar{\theta})\leq0,\forall i\in\{1,...,n\}$, we can combine the objective function with the contraints using the **Lagrange multipliers** \\(\lambda_1,...,\lambda_n\geq0\\). 

$$L(\bar{\theta},\bar{\lambda})=f(\theta)+\sum_{i=1}^n{\lambda_ih_i(\bar{\theta})}$$

From this formation, we observe that if a model satifies all the constraints, \\(f(\bar{\theta})\geq L(\bar{\theta},\bar{\lambda})\\), so minimizing \\(f\\) is the same as minimizing the maximum of \\(L\\), that is:

$$\displaystyle\min_{\bar{\theta}}\max_{\bar{\lambda},\lambda_i\geq0}{L(\bar{\theta},\bar{\lambda}})$$

This is called the **primal formulation**. And we have **dual formulation**:

$$\displaystyle\max_{\bar{\lambda},\lambda_i\geq0}\min_{\bar{\theta}}{L(\bar{\theta},\bar{\lambda}})$$

The dual provides a lower bound for the primal solution, so there is a **duality gap** between the two formulations. The gap is 0 if the [**Karush–Kuhn–Tucker**](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) (**KKT**) conditions are satisfied:

$$\begin{aligned}
\nabla_{\bar{\theta}}L(\bar{\theta},\bar{\lambda})&=\mathbf{0}&\text{(Stationarity)}\\
\nabla_{\bar{\lambda}}L(\bar{\theta},\bar{\lambda})&=\mathbf{0}&\text{(Stationarity)}\\
\lambda_ih_i(\bar{\theta})&=0&\text{(Complementary Slackness)}\\
h_i(\bar{\theta})&\leq0&\text{(Primal Feasibility)}\\
\lambda_i&\geq0&\text{(Dual Feasibility)}
\end{aligned}$$

For our hard-margin SVM, the gap is 0. The Lagrangian function is:

$$L(\bar{w},b,\bar{\alpha})=\frac{\left\|\bar{w}\right\|^2}{2}+\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b))}$$

To satisfy the **stationarity** conditions, we need the gradient with respect to \\(\bar{w}\\) and \\(b\\) to be 0:

$$\begin{aligned}
\displaystyle\nabla_{\bar{w}}L(\bar{w},b,\bar{\alpha})=\bar{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}=\mathbf{0}&\Rightarrow\bar{w}^\ast=\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}\\
\nabla_{b}L(\bar{w},b,\bar{\alpha})=-\sum_{i=1}^n{\alpha_iy^{(i)}}=0&\Rightarrow\sum_{i=1}^n{\alpha_iy^{(i)}}=0
\end{aligned}$$

Using the dual formation, our problem become:

$$\begin{aligned}
\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b}L(\bar{w},b,\bar{\alpha})=&\;\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b}\frac{\left\|\bar{w}\right\|^2}{2}+\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b))}\\
=&\;\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}(\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)})}\cdot(\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}})\\
&+\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\alpha_iy^{(i)}\sum_{j=1}^n{\alpha_jy^{(j)}\bar{x}^{(j)}}\cdot\bar{x}^{(i)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\
=&\;\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\\
&+\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\\
=&\;\max_{\bar{\alpha},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}
\end{aligned}$$

According to the **complementary slackness** condition \\(\alpha^\ast_i(1-y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast))=0\\):

$$\begin{aligned}
&\alpha^\ast_i>0\Rightarrow y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast)=1&\text{ (support vector)}\\
&\alpha^\ast_i=0\Rightarrow y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast)>1&\text{ (non-support vector)}
\end{aligned}$$

We can also compute the intercept \\(b\\) using the support vectors:

$$\begin{aligned}
&\forall\alpha_k>0,y^{(k)}(\bar{w}^\ast\cdot\bar{x}^{(k)}+b^\ast)=1\\
&\Rightarrow\bar{w}^\ast\cdot\bar{x}^{(k)}+b^\ast=y^{(k)}\\
&\Rightarrow b^\ast=y^{(k)}-\bar{w}^\ast\cdot\bar{x}^{(k)}
\end{aligned}$$

### Soft-Margin SVM
However the hard-margin SVM above has limitations. If the data is not linearly separable, the SVM algorithm may not work. Consider the following example:

<p align="center">
<img src="/ml-replica/assets/images/soft-margin.png" width=300/>
</p>

If we use hard-margin SVM, the fitted model will be highly affected by the single outlier red point. But if we allow some misclassification by adding in the **slack variables**, the final model may be more robust. The setup for a soft-margin SVM is:

$$\begin{aligned}
\displaystyle\min_{\bar{w},b,\bar{\xi}}\;\;&{\frac{ {\left\|\bar{w}\right\|}^2}{2}+C\sum_{i=1}^n{\xi_i}},\\
\text{ subject to }\;&\xi_i\geq 0,y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1-\xi_i,\forall i\in\{1,...n\}
\end{aligned}$$

The Lagrangian is:

$$\begin{aligned}
\displaystyle L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=\frac{\left\|\bar{w}\right\|^2}{2}+C\sum_{i=1}^n{\xi_i}+\sum_{i=1}^n{\alpha_i(1-\xi_i-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b))}+\sum_{i=1}^n{\beta_i(-\xi_i)}
\end{aligned}$$

We first find the gradient with respect to \\(\bar{w}\\), \\(b\\), \\(\bar{\xi}\\), and we need them to be 0:

$$\begin{aligned}
\nabla_{\bar{w}}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=&\;\bar{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}&&\Rightarrow\bar{w}^\ast=\sum_{i=1}^n{\alpha_i^\ast y^{(i)}\bar{x}^{(i)}}\\
\nabla_{b}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=&\;-\sum_{i=1}^n{\alpha_iy^{(i)}}&&\Rightarrow\sum_{i=1}^n{\alpha_i^\ast y^{(i)}}=0\\
\nabla_{\bar{\xi}}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=&\;
\begin{bmatrix} 
C-\alpha_1-\beta_1\\ 
\vdots\\
C-\alpha_n-\beta_n
\end{bmatrix}&&\Rightarrow\alpha_i^\ast=C-\beta_i^\ast\Rightarrow0\leq\alpha_i^\ast\leq C
\end{aligned}$$

And the dual formulation is:

$$\begin{aligned}
\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b,\bar{\xi}}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=&\;\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b,\bar{\xi}}\frac{\left\|\bar{w}\right\|^2}{2}+C\sum_{i=1}^n{\xi_i}\\
&+\sum_{i=1}^n{\alpha_i(1-\xi_i-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b))}+\sum_{i=1}^n{\beta_i(-\xi_i)}\\
=&\;\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}+\sum_{i=1}^n{(C-\alpha_i-\beta_i)\xi_i}\\
&+\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\
=&\;\max_{\bar{\alpha},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}
\end{aligned}$$

We can see that soft-margin SVM has a same dual formulation as the hard-margin SVM. And now, the condition for optimum are \\(\alpha_i^\ast(1-\xi_i^\ast y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast))=0\\) **AND** \\(\beta_i^\ast(-\xi_i^\ast)=0\\), so combining them together:

$$\begin{aligned}
&&\alpha^\ast_i=0&\Rightarrow\beta^\ast_i=C\Rightarrow\xi^\ast_i=0\\
&&&\Rightarrow y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast)\geq1-\xi^\ast_i=1&\text{ (non-support vector)}\\
&&\alpha^\ast_i=C&\Rightarrow\beta^\ast_i=0\Rightarrow\xi^\ast_i\geq0\\
&&&\Rightarrow y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast)=1-\xi_i^\ast\leq1&\text{ (support vector off the margin)}\\
&&0<\alpha_i^\ast<C&\Rightarrow0<\beta^\ast_i<C\Rightarrow\xi^\ast_i=0\\
&&&\Rightarrow y^{(i)}(\bar{w}^\ast\cdot\bar{x}^{(i)}+b^\ast)=1-\xi^\ast_i=1&\text{ (support vector on the margin)}
\end{aligned}$$

An observation that can be drawn from this result is that \\(C\\) is a hyperparameter that controls the "softness" of our SVM model. If \\(C\\) is big enough, the soft-margin SVM will become a hard-margin one.

### Feature Mapping
Soft-margin SVM seems to provide a decent approach to non-linearly separable data, but it only works well when there are a few "noisy" data. When the boundary between categories is inherently non-linear, it is not reasonable to use a soft-margin SVM. For [example](http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html):

<p align="center">
<img src="/ml-replica/assets/images/non-sep.png" width=300/>
</p>

We cannot find a separating hyperplane, so the SVMs we have discussed will not work as expected. However, if we jump out of the 2-dimensional space, we can find hyperplanes that can separate the data:

<p align="center">
<img src="/ml-replica/assets/images/feature-map.png" width=350/>
</p>

The way we convert the lower-dimensional coordinates into higher ones is called a **feature mapping**.

### Kernel Trick
Assume we have a feature mapping \\(\phi(\cdot):\mathcal{X}\rightarrow\mathcal{F}\\) and we fit this mapped data using SVM, then the objective function would be:

$$\begin{aligned}
J(\bar{\alpha})=\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\left<\phi(\bar{x}^{(i)}),\phi(\bar{x}^{(j)})\right>}}
\end{aligned}$$

One problem with this setup is that the computation can be slow because we need to 1) map the features to higher dimension, 2) compute the inner products between each pair of mapped features. Also, note that predicting the response for a new data point is:

$$\begin{aligned}
\hat{y}&=\text{sgn}((\bar{w}^\ast)^\mathsf{T}\phi(\hat{\bar{x}})+b^\ast)=\text{sgn}(\sum_{i=1}^n{\alpha_iy^{(i)}\left<\phi(\bar{x}^{(i)}),\phi(\hat{\bar{x}})\right>}+b^\ast)\\
b^\ast&=y^{(k)}-(\bar{w}^\ast)^\mathsf{T}\phi(\bar{x}^{(k)})=y^{(k)}-\sum_{i=1}^n{\alpha_iy^{(i)}\left<\phi(\bar{x}^{(i)}),\phi(\bar{x}^{(k)})\right>},\forall\alpha_k>0
\end{aligned}$$

We can see that only the inner product of the mappings are needed in training or evaluation. So instead of computing the mapping, we would like to compute the inner products of the mapped features directly. Therefore, we introduce the **kernel function**:

$$\begin{aligned}
&K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}\\
&\text{ s.t. }\exists\phi:\mathcal{X}\rightarrow\mathcal{F},\forall \bar{x},\bar{x}'\in\mathcal{X}\Rightarrow K(\bar{x},\bar{x}')=\left<\phi(\bar{x}),\phi(\bar{x}')\right>
\end{aligned}$$

Then we can rewrite the objective function as:

$$\begin{aligned}
J(\bar{\alpha})=\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})}}
\end{aligned}$$

According to [**Mercer's Theorem**](https://xavierbourretsicotte.github.io/Kernel_feature_map.html#Necessary-and-sufficient-conditions), a kernel function is valid if and only if its **Gram matrix** must be positive semi-definite. Below are some properties of kernel functions, let \\(K_1,K_2:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}\\) be two valid kernels of feature mapping \\(\phi_1:\mathcal{X}\rightarrow\mathbb{R}^{M_1},\phi_2:\mathcal{X}\rightarrow\mathbb{R}^{M_2}\\), then the following kernels and feature maps \\(K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R},\phi:\mathcal{X}\rightarrow\mathcal{F}\\) are valid (\\(\phi\\) is not unique):

$$\begin{aligned}
K(\bar{u},\bar{v})&=\alpha K_1(\bar{u},\bar{v}),\alpha>0,\\
\phi(\bar{x})&=\alpha\phi_1(\bar{x})\\
K(\bar{u},\bar{v})&=f(\bar{u})K_1(\bar{u},\bar{v})f(\bar{v}),\forall f:\mathcal{X}\rightarrow\mathbb{R},\\
\phi(\bar{x})&=f(\bar{x})\phi(\bar{x})\\
K(\bar{u},\bar{v})&=K_1(\bar{u},\bar{v})+K_2(\bar{u},\bar{v}),\phi(\bar{x})\in\mathbb{R}^{M_1+M_2},\\
\phi(\bar{x})&=
\begin{bmatrix*}[l]
\phi_1(\bar{x})^{(1)}\\
\;\;\;\;\;\;\vdots\\
\phi_1(\bar{x})^{(M_1)}\\
\phi_2(\bar{x})^{(1)}\\
\;\;\;\;\;\;\vdots\\
\phi_2(\bar{x})^{(M_2)}
\end{bmatrix*}\\
K(\bar{u},\bar{v})&=K_1(\bar{u},\bar{v})K_2(\bar{u},\bar{v}),\phi(\bar{x})\in\mathbb{R}^{M_1M_2},\\
\phi(\bar{x})&=
\begin{bmatrix*}[c]
\phi_1(\bar{x})^{(1)}\phi_2(\bar{x})^{(1)}\\
\phi_1(\bar{x})^{(1)}\phi_2(\bar{x})^{(2)}\\
\vdots\\
\phi_1(\bar{x})^{(M_1)}\phi_2(\bar{x})^{(M_2-1)}\\
\phi_1(\bar{x})^{(M_1)}\phi_2(\bar{x})^{(M_2)}
\end{bmatrix*}
\end{aligned}$$

Using these properties, we can come up with some useful kernel functions:

$$\begin{aligned}
K(\bar{u},\bar{v})&=\bar{u}\cdot\bar{v}&\text{ (Linear Kernel)}\\
K(\bar{u},\bar{v})&=(\bar{u}\cdot\bar{v}+1)^p&\text{ (Polynomial Kernel)}\\
K(\bar{u},\bar{v})&=e^{-\gamma\left\|\bar{u}-\bar{v}\right\|^2}&\text{ (RBF Kernel)}\\
\end{aligned}$$

While the linear and polynomial kernels may be obvious (use the addition and product rule), the RBF kernel can be hard to interpret:

$$\begin{aligned}
K(\bar{u},\bar{v})&=e^{-\gamma\left\|\bar{u}-\bar{v}\right\|^2}\\
&=e^{-\gamma\left\|\bar{u}\right\|^2-\gamma\left\|\bar{v}\right\|^2+2\gamma\bar{u}\cdot\bar{v}}\\
&=e^{-\gamma\left\|\bar{u}\right\|^2}e^{2\gamma\bar{u}\cdot\bar{v}}e^{-\gamma\left\|\bar{v}\right\|^2}
\end{aligned}$$

Now this looks like the second transformation above, we would like to prove the middle term a kernel. We will use Taylor expansion:

$$\begin{aligned}
K(\bar{u},\bar{v})&=e^{2\gamma\bar{u}\cdot\bar{v}}\\
&=\frac{(2\gamma\bar{u}\cdot\bar{v})^0}{0!}+\frac{(2\gamma\bar{u}\cdot\bar{v})^1}{1!}+...+\frac{(2\gamma\bar{u}\cdot\bar{v})^n}{n!}+...
\end{aligned}$$

So the middle term is in fact an infinite positive-weighted-sum of polynomial kernels, which is also a valid kernel. And we can tell that the feature mapping of a RBF kernel will have infinite dimensions, so it proves the importance of a kernel function as calculating the mapped features can be impossible.

### Sequential Minimal Optimization
*Reference*: [John C. Platt, Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines](https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf)

Now the only thing we need is to pick the multipliers to optimize the objective function. In another word, we are solving this **Quadratic Programming** problem:

$$\begin{aligned}
\max_{\bar{\alpha}}\;\;&{\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})}}}\\
\text{subject to }\;&0\leq\alpha_i\leq C, \forall i = 1...n,\\
&\sum_{i=1}^n{\alpha_i y^{(i)}}=0
\end{aligned}$$

The main idea of the **Sequential Minimal Optimization (SMO)** algorithm is to optimize only a **pair** of multipliers each time. It works as following:

*Pseudocode*:
<pre>
<b>α</b>=0, b=0 
<b>while</b> not all α satisfies <b>KKT</b> conditions:
    <b>pick</b> αi, αj using some <b>heuristics</b>
    <b>optimize</b> αi, αj
    <b>update</b> b
</pre>

The optimization for each pair can be represented as (\\(\delta\\) is the sum of the rest terms):

$$\begin{aligned}
\max_{\alpha_i,\alpha_j}\;\;&\alpha_i+\alpha_j-\frac{1}{2}\alpha_i^2K(\bar{x}^{(i)},\bar{x}^{(i)})-\frac{1}{2}\alpha_j^2K(\bar{x}^{(j)},\bar{x}^{(j)})\\
&-\alpha_iy^{(i)}\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}\\
&-\alpha_jy^{(j)}\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(j)})}\\
&-\alpha_i\alpha_jy^{(i)}y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})+\delta\\
\text{subject to}\;\;&0\leq\alpha_i,\alpha_j\leq C\\
&\alpha_iy^{(i)}+\alpha_jy^{(j)}=-\sum_{1\leq q\leq n\atop q\neq i,j}\alpha_qy^{(q)}=\zeta
\end{aligned}$$

Now we can substitute \\(\alpha_j\\) for \\(\alpha_i\\):

$$
\begin{aligned}
&\alpha_iy^{(i)}+\alpha_jy^{(j)}=\zeta\\
&\begin{aligned}
\Rightarrow\;\;\;\;&&\alpha_i=   &\;\zeta y^{(i)}-\alpha_j y^{(i)}y^{(j)}\\
\Rightarrow\;\;\;\;&&J(\alpha_j)=&\;\zeta y^{(i)}-\alpha_j y^{(i)}y^{(j)}+\alpha_j\\
                   &&            &-\frac{1}{2}(\zeta-\alpha_j y^{(j)})^2K(\bar{x}^{(i)},\bar{x}^{(i)})-\frac{1}{2}\alpha_j^2K(\bar{x}^{(j)},\bar{x}^{(j)})\\
                   &&            &-(\zeta-\alpha_jy^{(j)})S_i-\alpha_jy^{(j)}S_j\\
                   &&            &-(\zeta-\alpha_j y^{(j)})\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})+\delta\\
       \text{where}&&S_i=        &\;\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}\\
                   &&S_j=        &\;\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(j)})}
\end{aligned}
\end{aligned}
$$

To optimize, we take the partial derivative w/ respect to \\(\alpha_j\\):

$$\begin{aligned}
\frac{\partial J(\alpha_j)}{\partial\alpha_j}=&\;\alpha_j(2K(\bar{x}^{(i)},\bar{x}^{(j)})-K(\bar{x}^{(i)},\bar{x}^{(i)})-K(\bar{x}^{(j)},\bar{x}^{(j)}))\\
&+\zeta y^{(j)}(K(\bar{x}^{(i)},\bar{x}^{(i)})-K(\bar{x}^{(i)},\bar{x}^{(j)}))\\
&+y^{(j)}(S_i-S_j)-y^{(i)}y^{(j)}+1\end{aligned}$$

If we look at the two sum terms \\(S_i\\), \\(S_j\\):

$$\begin{aligned}
S_i&=\sum_{0\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}\\
&=\sum_{q=0}^n{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}-\alpha_iy^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\
&=\bar{w}\cdot\phi(\bar{x}^{(i)})-\alpha_iy^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\
&=f(\bar{x}^{(i)})-b-(\zeta-\alpha_jy^{(j)})K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\
S_j&=f(\bar{x}^{(j)})-b-\alpha_jy^{(j)}K(\bar{x}^{(j)},\bar{x}^{(j)})-(\zeta-\alpha_jy^{(j)})K(\bar{x}^{(i)},\bar{x}^{(j)})
\end{aligned}$$

We want to derive the optimized \\(\alpha_j\\) by making the derivative to 0, assume we are currently at step \\(k\\):

$$\begin{aligned}
y^{(j)}(S_i^k-S_j^k)=&\;y^{(j)}(f^{k}(\bar{x}^{(i)})-f^{k}(\bar{x}^{(j)}))-\zeta y^{(j)}(K(\bar{x}^{(i)},\bar{x}^{(i)})-K(\bar{x}^{(i)},\bar{x}^{(j)}))\\
&+\alpha_j^{k}(K(\bar{x}^{(i)},\bar{x}^{(i)})+K(\bar{x}^{(j)},\bar{x}^{(j)})-2K(\bar{x}^{(i)},\bar{x}^{(j)}))\\
\frac{\partial J(\alpha_j)}{\partial\alpha_j}\big\rvert_{\alpha_j=\alpha_j^{k+1}}=&\;(\alpha_j^{k}-\alpha_j^{k+1})(K(\bar{x}^{(i)},\bar{x}^{(i)})+K(\bar{x}^{(j)},\bar{x}^{(j)})-2K(\bar{x}^{(i)},\bar{x}^{(j)}))\\
&+y^{(j)}((f^{k}(\bar{x}^{(i)})-y^{(i)})-(f^{k}(\bar{x}^{(j)})-y^{(j)}))=0\\
\alpha_j^{k+1}=&\;\alpha_j^{k}+\frac{y^{(j)}(E_i^k-E_j^k)}{\eta}\\
\text{where }E_i^k,E_j^k&\text{ are the residuals, }\eta=K(\bar{x}^{(i)},\bar{x}^{(i)})+K(\bar{x}^{(j)},\bar{x}^{(j)})-2K(\bar{x}^{(i)},\bar{x}^{(j)})\end{aligned}$$

Therefore, we are able to use the residual values and the kernel function to calculate the optimized \\(\alpha_j\\), and thus \\(\alpha_i\\) and \\(b\\). But before that, we need to check the constraints \\(0\leq\alpha_i^{k+1},\alpha_j^{k+1}\leq C\\). Since \\(\alpha_i^{k+1}y^{(i)}+\alpha_j^{k+1}y^{(j)}=\alpha_i^{k}y^{(i)}+\alpha_j^{k}y^{(j)}=\zeta\\):

$$
\begin{aligned}
y^{(i)}=y^{(j)}\Rightarrow&\;\alpha_i^{k+1}+\alpha_j^{k+1}=\alpha_i^{k}+\alpha_j^{k}\Rightarrow&\;\alpha_i^{k}+\alpha_j^{k}-C\leq&\;\alpha_j^{k+1}\leq\alpha_i^{k}+\alpha_j^{k}\\
y^{(i)}\neq y^{(j)}\Rightarrow&\;\alpha_i^{k+1}-\alpha_j^{k+1}=\alpha_i^{k}-\alpha_j^{k}\Rightarrow&\;\alpha_j^{k}-\alpha_i^{k}\leq&\;\alpha_j^{k+1}\leq C-\alpha_i^{k}+\alpha_j^{k}
\end{aligned}
$$

Therefore, we can get the lower and upper bounds of \\(\alpha_j^{k+1}\\), \\(L^{k+1}\\) and \\(H^{k+1}\\), based on the value of \\(y^{(i)},y^{(j)}\\), and "clip" the value of \\(\alpha_j^{k+1}\\):

$$
\begin{aligned}
L^{k+1}&=\begin{cases}
\max(0, \alpha_i^{k}+\alpha_j^{k}-C)&\text{if }y^{(i)}=y^{(j)}\\
\max(0, \alpha_j^{k}-\alpha_i^{k})&\text{otherwise}
\end{cases}\\
H^{k+1}&=\begin{cases}
\min(C, \alpha_i^{k}+\alpha_j^{k})&\text{if }y^{(i)}=y^{(j)}\\
\min(C, C-\alpha_i^{k}+\alpha_j^{k})&\text{otherwise}
\end{cases}\\
\alpha_j^{k+1}&=\min(\max(\alpha_j^{k+1}, L^{k+1}), H^{k+1})
\end{aligned}
$$

And we can get \\(\alpha_i^{k+1}\\):

$$\begin{aligned}
\alpha_i^{k+1}=&\;\frac{\alpha_i^{k}y^{(i)}+\alpha_j^{k}y^{(j)}-\alpha_j^{k+1}y^{(j)}}{y^{(i)}}=\alpha_i^{k}+y^{(i)}y^{(j)}(\alpha_j^{k}-\alpha_j^{k+1})\\
\end{aligned}
$$

Now we want to get the offset value \\(b^{k+1}\\) by using the support vectors:

$$
\begin{aligned}
&0<\alpha_i^{k+1}<C\\
&\Rightarrow\alpha_i^{k+1}\text{ on the margin }\\
                  &\Rightarrow y^{(i)}(\bar{w}\cdot\phi(\bar{x}^{(i)})+b^{k+1})=1\\
                  &\;\begin{aligned}
                  \Rightarrow b_i^{k+1}=&\;y^{(i)}-(f^k(\bar{x}^{(i)})-b^{k}-\alpha_i^{k}y^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_j^{k}y^{(i)}K(\bar{x}^{(j)},\bar{x}^{(j)})\\
                  &+\alpha_i^{k+1}y^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_j^{k+1}y^{(i)}K(\bar{x}^{(j)},\bar{x}^{(j)}))\\
                                     =&\;b^{k}-E_i^{k}+(\alpha_i^{k}-\alpha_i^{k+1})y^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})+(\alpha_j^{k}-\alpha_j^{k+1})y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})
                  \end{aligned}\\
&0<\alpha_j^{k+1}<C\\
&\Rightarrow\alpha_j^{k+1}\text{ on the margin }\\
                  &\Rightarrow b_j^{k+1}=b^{k}-E_j^{k}+(\alpha_j^{k}-\alpha_j^{k+1})y^{(j)}K(\bar{x}^{(j)},\bar{x}^{(j)})+(\alpha_i^{k}-\alpha_i^{k+1})y^{(i)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\
\end{aligned}
$$

If \\(\alpha_i^{k+1}\\) is on the margin, we will set \\(b^{k+1}=b_i^{k+1}\\); if \\(\alpha_j^{k+1}\\) is on the margin, \\(b^{k+1}=b_j^{k+1}\\); if they are both on the margin, \\(b^{k+1}=b_i^{k+1}=b_j^{k+1}\\). But if none of them is on the margin, the SMO algorithm chooses the average value \\((b^{k+1}=b_i^{k+1}+b_j^{k+1})/2\\).

Now we are able to optimize any pair of multipliers, \\(\alpha_i^{k+1}\\) and \\(\alpha_j^{k+1}\\), and get the updated offset value, \\(b^{k+1}\\). But we have not talked about the mechanism of choosing the pair, \\(\alpha_i^{k}, \alpha_j^{k}\\). In Platt's original paper, two heuristics are used for \\(\alpha_i\\) and \\(\alpha_j\\) respectively:

- For \\(\alpha_i\\):
    
    If all \\(\alpha=0\\) or \\(C\\), choose \\(\alpha_i\\) that violates the KKT conditions<br>
    Otherwise choose \\(0<\alpha_i< C\\) that violates the KKT conditions<br>

- For \\(\alpha_j\\):

    Choose \\(\alpha_j\\) that maximize \\(\|E_i-E_j\|\\). The algorithm will keep track of \\(E_k\\) for all \\(0<\alpha_k< C\\)

Note we only need to check the **complementary slackness** conditions here because the other conditions are enforced by the SMO algorithm. A full pseudocode can be found in Platt's paper.

Alternatively, we can give up the heuristics and instead randomly select \\(\alpha_j\\) for each \\(\alpha_i\\). This [Simplified SMO](http://cs229.stanford.edu/materials/smo.pdf) is much easier to implement:

*Pseudocode*:
<pre>
<b>α</b>=<b>0</b>, b=0
tol=1e-3, ε=1e-5
iter=0
<b>while</b> iter<max_iter: // stop when max_iter pass of no updates
    k = 0 // number of α changed
    <b>for</b> i in 1...n:
        Ei=f(xi)-yi
        <b>if</b> (yiEi<-tol <b>and</b> αi<C) <b>or</b> 
                (yiEi>tol <b>and</b> αi>0): // KKT condition
            <b>random</b> j!=i
            Ej=f(xj)-yj
            <b>compute</b> L, H
            <b>if</b> L==H:
                <b>continue</b>
            η=K(xi,xi)+K(xj,xj)-2K(xi,xj)
            <b>if</b> η>=0:
                <b>continue</b>
            αj'=αj+yj(Ei-Ej)/η
            <b>clip</b> αj' with L, H
            <b>if</b> |αj-αj'|<ε:
                <b>continue</b>
            αi'=αi+yiyj(αj-αj')
            <b>compute</b> bi, bj
            <b>if</b> 0<αi'<C:
                b'=bi
            <b>elif</b> 0<αj'<C:
                b'=bj
            <b>else</b>:
                b'=(bi+bj)/2
            αi=αi', αj=αj', b=b'
            k++
    <b>if</b> k==0:
        iter+=1 // can't optimize w/ αi
    <b>else</b>:
        iter=0
</pre>

Code: [_svm.py](https://github.com/xianglous/ml-replica/tree/main/mlreplica/linear_model/_svm.py)

# Regression

In classification problems, we try to classify input features into categories. But we may want to predict more diverse and continuous data. So instead of finding a function \\(f:\mathbb{R}^d\rightarrow\\{-1,1\\}\\), we want to model the function \\(f:\mathbb{R}^d\rightarrow\mathbb{R}\\).

## Ordinary Least Squares

Previously, we have been using the **Hinge Loss** as the objective for optimization because it takes the **product**, \\(y\hat{y}\\), as a measure of the similarity between the fitted value and the actual class and penalize it by substract it from 1 (100% similarity). However, the goal of regression is to predict a value rather than a class, so our loss function should somehow measure the **distance** between the predicted and actual value. For example, we can use the **Squared Loss**:

$$h(y,\hat{y}) = \frac{(y-\hat{y})^2}{2}$$

And for the entire training data, we measure the **Mean Squared Error** (MSE):

$$L(\bar{y}, \hat{\bar{y}})=\frac{1}{n}\sum_{i=1}^n{\frac{(y^{(i)}-\hat{y}^{(i)})^2}{2}}=\frac{1}{2n}(\bar{y}-\hat{\bar{y}})^\mathsf{T}(\bar{y}-\hat{\bar{y}})$$

So a common method for solving the linear regression problem is **Ordinary Least Squares** (OLS). We want to minimize the loss:

$$L(X, \bar{y}, \bar{w})=\frac{1}{n}\sum_{i=1}^n{\frac{(y^{(i)}-\bar{w}\cdot\bar{x}^{(i)})^2}{2}}=\frac{1}{2n}(\bar{y}-X\bar{w})^\mathsf{T}(\bar{y}-X\bar{w})$$

We would like the gradient to shrink to 0:

$$
\begin{aligned}
\nabla_{\bar{w}}L(X, \bar{y}, \bar{w})=&\frac{1}{2n}(2X^\mathsf{T} X\bar{w}-2X^\mathsf{T}\bar{y})=\frac{1}{n}(X^\mathsf{T} X\bar{w}-X^\mathsf{T}\bar{y})\\
\nabla_{\bar{w}^*}L(X, \bar{y}, \bar{w})=&\frac{1}{n}(X^\mathsf{T} X\bar{w}^*-X^\mathsf{T}\bar{y})=\mathbb{0}\Rightarrow X^\mathsf{T} X\bar{w}^*=X^\mathsf{T}\bar{y}
\end{aligned}
$$

So, if \\(X^\mathsf{T} X\\) is a **Nonsingular** matrix, we can find a **closed-form solution** to the OLS regression problem:

$$\bar{w}^*=(X^\mathsf{T} X)^{-1}X^\mathsf{T}\bar{y}$$

But computing such matrix operations can be expensive when the dimensions of \\(X\\) is big. So as usual, we can use **gradient descent**.

*Pseudocode (SGD)*:
<pre>
k=0, w=0
<b>while</b> criterion not met:
    <b>for</b> i in 1...n:
        w = w - ηxi(w•xi-yi) // the Descent
        k++
</pre>

## Regularization

### Ridge Regression

### Lasso Regression
