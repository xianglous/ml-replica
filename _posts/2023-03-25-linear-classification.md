---
title: Linear Classification
layout: post
---

# Table of Content
- [Table of Content](#table-of-content)
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


# Linear Clasification
Linear classifiers classifies the input features based on the decision hyperplane in the feature space.

## Perceptron
The perceptron algorithm is the building block of deep learning. It updates on one data point at each time and moves in the "right" direction based on that point. <br><br>
*Pseudocode*

(w/o offset)
<pre>
k=0, w=0
<b>while</b> not all correctly classified <b>and</b> k < max step:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w•Xi) <= 0: // misclassified
            w = w + yiXi // the Update
            k++
</pre>
(w/ offset)
<pre>
k=0, w=0, b=0
<b>while</b> not all correctly classified <b>and</b> k < max step:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w•Xi+b) <= 0: // misclassified
            w = w + yiXi // the Update
            b = b + yi // offset update
            k++
</pre>
*Note*: we can convert the w/ offset version to w/o by transforming `X` to `[1, X]`, then the first resulting weight parameter would be the offset.

*Code*: [algorithm.py - perceptron()](https://github.com/xianglous/ml-replica/tree/main/mlreplica/utils/algorithm.py#L6)

## Stochastic Gradient Descent
Perceptron is nice and simple, but it has an important restriction: it only converges on **linearly-separable** data. To make it work for non-separable data, we need to change the way it approaches the best model. 

### Loss Functions
A loss function measures the fit of the current model to the training data. For the perceptron algorithm, we want our fitted value to have the same sign as the actual class, so we multiply them together and penalize those with negative products. In another word, the perceptron algorithm uses the following loss function:

$$L(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}\sum_{i=1}^n{[y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)})\leq 0]}$$

A problem with this loss function is that it does not measures the **distance** between the predicted value and actual class, so 0.1 and 1 will all be seen as good classifications while -0.1 and -1 will all be equally bad. <br>

So another loss function we can use instead is the **Hinge Loss**, for a fitted value \\(\hat{y}\\), the Hingle Loss is:

$$h(y,\hat{y})=\max(0, 1-y\hat{y})$$

For our linear model, the loss function is defined as:

$$L(\mathbf{X}, \mathbf{y}, \mathbf{w})=\frac{1}{n}\sum_{i=1}^n{\max(0, 1-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)})))}$$

This loss will penalize any **imperfect** prediction.

### Gradient Descent
The loss function tells us about how **bad** the current model fits the data. Therefore, we need to know the direction in which moving the parameters will decrease the loss. In mathematics, we use the gradient to measure the "direction." For Hinge Loss, the gradient of a single data point is: 

$$\nabla_{\mathbf{w}}{h(\mathbf{x}^{(i)}, y^{(i)},\mathbf{w})}=
    \begin{cases}
    -y^{(i)}\mathbf{x}^{(i)}&\text{if }y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)})<1\\
    \mathbf{0} & \text{otherwise}
    \end{cases}$$

And the gradient of the whole training data is:

$$\nabla_{\mathbf{w}}{L(\mathbf{X}, \mathbf{y},\mathbf{w})}=\frac{1}{n}\sum_{i=1}^n{\nabla_{\mathbf{w}}{h(\mathbf{x}^{(i)}, y^{(i)},\mathbf{w})}}$$

By moving the the weights in the direction of the gradient, we will likely decrease the loss of our model. So the **Gradient Descent** is defined as:

$$\mathbf{w}^{(k+1)}=\mathbf{w}^{(k)}-\eta\nabla_{\mathbf{w}}{L(\mathbf{X}, \mathbf{y}, \mathbf{w})}$$

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
<img src=" /ml-replica/assets/img/non_max_margin.png" width=300/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src=" /ml-replica/assets/img/max_margin.png" width=300/>
</p>

The solid lines on each figure represents a classification boundary that separates the two training classes. They both perfectly classify the training data, but which one is better? We would consider the right one better because it maximizes the "margins" between the separator and the training data.<br>
So while using the Hinge Loss may produce either of the above models, maximizing the margin will give us the better model. In another word, we want to maximize the distance from the closest point to the separating hyperplane:

<p align="center">
<img src=" /ml-replica/assets/img/max_margin_distance.png" width=300/>
</p>

If we look at the two margine lines, they are actually the decision lines \\(\mathbf{w}\cdot\mathbf{x}+b=1\\) and \\(\mathbf{w}\cdot\mathbf{x}+b=-1\\) beecause they are right on the border of being penalized. So we can calculate the margin as the distance between the positive margin line and the decision boundary:

$$d=\frac{|(1-b)-(-b)|}{\left\|\mathbf{w}\right\|}=\frac{1}{\left\|\mathbf{w}\right\|}$$

And we want our model to maximize the margin:

$$\displaystyle\max_{\mathbf{w}, b}{\frac{1}{\left\|\mathbf{w}\right\|}}$$

### Hard-Margin SVM
We can now formulate our problem as a constrained optimization. For computation purpose, we transform the maximization into a minimization problem:

$$\begin{aligned}
\displaystyle\min_{\mathbf{w}}\;\;&{\frac{ {\left\|\mathbf{w}\right\|}^2}{2}}\\
\text{ subject to }\;&y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b)\geq1,\forall i\in\{1,...n\}
\end{aligned}$$

### Lagrange Duality
For a constrained optimization problem \\(\min_{\textbf{θ}}{f(\textbf{θ})}\\) subject to \\(\eta\\) constraints \\(h_i(\textbf{θ})\leq 0,\forall i\in\{1,...,n\}\\), we can combine the objective function with the contraints using the **Lagrange multipliers** \\(\lambda_1,...,\lambda_n\geq0\\). 

$$L(\textbf{θ},\textbf{λ})=f(\theta)+\sum_{i=1}^n{\lambda_ih_i(\textbf{θ})}$$

From this formation, we observe that if a model satifies all the constraints, \\(f(\textbf{θ})\geq L(\textbf{θ},\textbf{λ})\\), so minimizing \\(f\\) is the same as minimizing the maximum of \\(L\\), that is:

$$\displaystyle\min_{\textbf{θ}}\max_{\textbf{λ},\lambda_i\geq0}{L(\textbf{θ},\textbf{λ}})$$

This is called the **primal formulation**. And we have **dual formulation**:

$$\displaystyle\max_{\textbf{λ},\lambda_i\geq0}\min_{\textbf{θ}}{L(\textbf{θ},\textbf{λ}})$$

The dual provides a lower bound for the primal solution, so there is a **duality gap** between the two formulations. The gap is 0 if the [**Karush–Kuhn–Tucker**](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) (**KKT**) conditions are satisfied:

$$\begin{aligned}
\nabla_{\textbf{θ}}L(\textbf{θ},\textbf{λ})&=\mathbf{0}&\text{(Stationarity)}\\
\nabla_{\textbf{λ}}L(\textbf{θ},\textbf{λ})&=\mathbf{0}&\text{(Stationarity)}\\
\lambda_ih_i(\textbf{θ})&=0&\text{(Complementary Slackness)}\\
h_i(\textbf{θ})&\leq0&\text{(Primal Feasibility)}\\
\lambda_i&\geq0&\text{(Dual Feasibility)}
\end{aligned}$$

For our hard-margin SVM, the gap is 0. The Lagrangian function is:

$$L(\mathbf{w},b,\textbf{α})=\frac{\left\|\mathbf{w}\right\|^2}{2}+\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b))}$$

To satisfy the **stationarity** conditions, we need the gradient with respect to \\(\mathbf{w}\\) and \\(b\\) to be 0:

$$\begin{aligned}
\displaystyle\nabla_{\mathbf{w}}L(\mathbf{w},b,\textbf{α})=\mathbf{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\mathbf{x}^{(i)}}=\mathbf{0}&\Rightarrow\mathbf{w}^\ast=\sum_{i=1}^n{\alpha_iy^{(i)}\mathbf{x}^{(i)}}\\
\nabla_{b}L(\mathbf{w},b,\textbf{α})=-\sum_{i=1}^n{\alpha_iy^{(i)}}=0&\Rightarrow\sum_{i=1}^n{\alpha_iy^{(i)}}=0
\end{aligned}$$

Using the dual formation, our problem become:

$$\begin{aligned}
\max_{\textbf{α},\alpha_i\geq0}\min_{\mathbf{w},b}L(\mathbf{w},b,\textbf{α})=&\;\max_{\textbf{α},\alpha_i\geq0}\min_{\mathbf{w},b}\frac{\left\|\mathbf{w}\right\|^2}{2}+\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b))}\\
=&\;\max_{\textbf{α},\alpha_i\geq0}\frac{1}{2}(\sum_{i=1}^n{\alpha_iy^{(i)}\mathbf{x}^{(i)})}\cdot(\sum_{i=1}^n{\alpha_iy^{(i)}\mathbf{x}^{(i)}})\\
&+\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\alpha_iy^{(i)}\sum_{j=1}^n{\alpha_jy^{(j)}\mathbf{x}^{(j)}}\cdot\mathbf{x}^{(i)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\
=&\;\max_{\textbf{α},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\mathbf{x}^{(i)}}\cdot\mathbf{x}^{(j)}}\\
&+\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\mathbf{x}^{(i)}}\cdot\mathbf{x}^{(j)}}\\
=&\;\max_{\textbf{α},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\mathbf{x}^{(i)}}\cdot\mathbf{x}^{(j)}}
\end{aligned}$$

According to the **complementary slackness** condition \\(\alpha^\ast_i(1-y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast))=0\\):

$$\begin{aligned}
&\alpha^\ast_i>0\Rightarrow y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast)=1&\text{ (support vector)}\\
&\alpha^\ast_i=0\Rightarrow y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast)>1&\text{ (non-support vector)}
\end{aligned}$$

We can also compute the intercept \\(b\\) using the support vectors:

$$\begin{aligned}
&\forall\alpha_k>0,y^{(k)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(k)}+b^\ast)=1\\
&\Rightarrow\mathbf{w}^\ast\cdot\mathbf{x}^{(k)}+b^\ast=y^{(k)}\\
&\Rightarrow b^\ast=y^{(k)}-\mathbf{w}^\ast\cdot\mathbf{x}^{(k)}
\end{aligned}$$

### Soft-Margin SVM
However the hard-margin SVM above has limitations. If the data is not linearly separable, the SVM algorithm may not work. Consider the following example:

<p align="center">
<img src="/ml-replica/assets/img/soft-margin.png" width=300/>
</p>

If we use hard-margin SVM, the fitted model will be highly affected by the single outlier red point. But if we allow some misclassification by adding in the **slack variables**, the final model may be more robust. The setup for a soft-margin SVM is:

$$\begin{aligned}
\displaystyle\min_{\mathbf{w},b,\textbf{ξ}}\;\;&{\frac{ {\left\|\mathbf{w}\right\|}^2}{2}+C\sum_{i=1}^n{\xi_i}},\\
\text{ subject to }\;&\xi_i\geq 0,y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b)\geq1-\xi_i,\forall i\in\{1,...n\}
\end{aligned}$$

The Lagrangian is:

$$\begin{aligned}
\displaystyle L(\mathbf{w},b,\textbf{ξ},\textbf{α},\textbf{β})=\frac{\left\|\mathbf{w}\right\|^2}{2}+C\sum_{i=1}^n{\xi_i}+\sum_{i=1}^n{\alpha_i(1-\xi_i-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b))}+\sum_{i=1}^n{\beta_i(-\xi_i)}
\end{aligned}$$

We first find the gradient with respect to \\(\mathbf{w}\\), \\(b\\), \\(\textbf{ξ}\\), and we need them to be 0:

$$\begin{aligned}
\nabla_{\mathbf{w}}L(\mathbf{w},b,\textbf{ξ},\textbf{α},\textbf{β})=&\;\mathbf{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\mathbf{x}^{(i)}}&&\Rightarrow\mathbf{w}^\ast=\sum_{i=1}^n{\alpha_i^\ast y^{(i)}\mathbf{x}^{(i)}}\\
\nabla_{b}L(\mathbf{w},b,\textbf{ξ},\textbf{α},\textbf{β})=&\;-\sum_{i=1}^n{\alpha_iy^{(i)}}&&\Rightarrow\sum_{i=1}^n{\alpha_i^\ast y^{(i)}}=0\\
\nabla_{\textbf{ξ}}L(\mathbf{w},b,\textbf{ξ},\textbf{α},\textbf{β})=&\;
\begin{bmatrix} 
C-\alpha_1-\beta_1\\ 
\vdots\\
C-\alpha_n-\beta_n
\end{bmatrix}&&\Rightarrow\alpha_i^\ast=C-\beta_i^\ast\Rightarrow0\leq\alpha_i^\ast\leq C
\end{aligned}$$

And the dual formulation is:

$$\begin{aligned}
\max_{\textbf{α},\alpha_i\geq0}\min_{\mathbf{w},b,\textbf{ξ}}L(\mathbf{w},b,\textbf{ξ},\textbf{α},\textbf{β})=&\;\max_{\textbf{α},\alpha_i\geq0}\min_{\mathbf{w},b,\textbf{ξ}}\frac{\left\|\mathbf{w}\right\|^2}{2}+C\sum_{i=1}^n{\xi_i}\\
&+\sum_{i=1}^n{\alpha_i(1-\xi_i-y^{(i)}(\mathbf{w}\cdot\mathbf{x}^{(i)}+b))}+\sum_{i=1}^n{\beta_i(-\xi_i)}\\
=&\;\max_{\textbf{α},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\mathbf{x}^{(i)}}\cdot\mathbf{x}^{(j)}}+\sum_{i=1}^n{(C-\alpha_i-\beta_i)\xi_i}\\
&+\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\mathbf{x}^{(i)}}\cdot\mathbf{x}^{(j)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\
=&\;\max_{\textbf{α},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\mathbf{x}^{(i)}}\cdot\mathbf{x}^{(j)}}
\end{aligned}$$

We can see that soft-margin SVM has a same dual formulation as the hard-margin SVM. And now, the condition for optimum are \\(\alpha_i^\ast(1-\xi_i^\ast y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast))=0\\) **AND** \\(\beta_i^\ast(-\xi_i^\ast)=0\\), so combining them together:

$$\begin{aligned}
&&\alpha^\ast_i=0&\Rightarrow\beta^\ast_i=C\Rightarrow\xi^\ast_i=0\\
&&&\Rightarrow y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast)\geq1-\xi^\ast_i=1&\text{ (non-support vector)}\\
&&\alpha^\ast_i=C&\Rightarrow\beta^\ast_i=0\Rightarrow\xi^\ast_i\geq0\\
&&&\Rightarrow y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast)=1-\xi_i^\ast\leq1&\text{ (support vector off the margin)}\\
&&0<\alpha_i^\ast<C&\Rightarrow0<\beta^\ast_i<C\Rightarrow\xi^\ast_i=0\\
&&&\Rightarrow y^{(i)}(\mathbf{w}^\ast\cdot\mathbf{x}^{(i)}+b^\ast)=1-\xi^\ast_i=1&\text{ (support vector on the margin)}
\end{aligned}$$

An observation that can be drawn from this result is that \\(C\\) is a hyperparameter that controls the "softness" of our SVM model. If \\(C\\) is big enough, the soft-margin SVM will become a hard-margin one.

### Feature Mapping
Soft-margin SVM seems to provide a decent approach to non-linearly separable data, but it only works well when there are a few "noisy" data. When the boundary between categories is inherently non-linear, it is not reasonable to use a soft-margin SVM. For [example](http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html):

<p align="center">
<img src="/ml-replica/assets/img/non-sep.png" width=300/>
</p>

We cannot find a separating hyperplane, so the SVMs we have discussed will not work as expected. However, if we jump out of the 2-dimensional space, we can find hyperplanes that can separate the data:

<p align="center">
<img src="/ml-replica/assets/img/feature-map.png" width=350/>
</p>

The way we convert the lower-dimensional coordinates into higher ones is called a **feature mapping**.

### Kernel Trick
Assume we have a feature mapping \\(\phi(\cdot):\mathcal{X}\rightarrow\mathcal{F}\\) and we fit this mapped data using SVM, then the objective function would be:

$$\begin{aligned}
J(\textbf{α})=\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\left<\phi(\mathbf{x}^{(i)}),\phi(\mathbf{x}^{(j)})\right>}}
\end{aligned}$$

One problem with this setup is that the computation can be slow because we need to 1) map the features to higher dimension, 2) compute the inner products between each pair of mapped features. Also, note that predicting the response for a new data point is:

$$\begin{aligned}
\hat{y}&=\text{sgn}((\mathbf{w}^\ast)^\mathsf{T}\phi(\hat{\mathbf{x}})+b^\ast)=\text{sgn}(\sum_{i=1}^n{\alpha_iy^{(i)}\left<\phi(\mathbf{x}^{(i)}),\phi(\hat{\mathbf{x}})\right>}+b^\ast)\\
b^\ast&=y^{(k)}-(\mathbf{w}^\ast)^\mathsf{T}\phi(\mathbf{x}^{(k)})=y^{(k)}-\sum_{i=1}^n{\alpha_iy^{(i)}\left<\phi(\mathbf{x}^{(i)}),\phi(\mathbf{x}^{(k)})\right>},\forall\alpha_k>0
\end{aligned}$$

We can see that only the inner product of the mappings are needed in training or evaluation. So instead of computing the mapping, we would like to compute the inner products of the mapped features directly. Therefore, we introduce the **kernel function**:

$$\begin{aligned}
&K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}\\
&\text{ s.t. }\exists\phi:\mathcal{X}\rightarrow\mathcal{F},\forall \mathbf{x},\mathbf{x}'\in\mathcal{X}\Rightarrow K(\mathbf{x},\mathbf{x}')=\left<\phi(\mathbf{x}),\phi(\mathbf{x}')\right>
\end{aligned}$$

Then we can rewrite the objective function as:

$$\begin{aligned}
J(\textbf{α})=\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})}}
\end{aligned}$$

According to [**Mercer's Theorem**](https://xavierbourretsicotte.github.io/Kernel_feature_map.html#Necessary-and-sufficient-conditions), a kernel function is valid if and only if its **Gram matrix** must be positive semi-definite. Below are some properties of kernel functions, let \\(K_1,K_2:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}\\) be two valid kernels of feature mapping \\(\phi_1:\mathcal{X}\rightarrow\mathbb{R}^{M_1},\phi_2:\mathcal{X}\rightarrow\mathbb{R}^{M_2}\\), then the following kernels and feature maps \\(K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R},\phi:\mathcal{X}\rightarrow\mathcal{F}\\) are valid (\\(\phi\\) is not unique):

$$\begin{aligned}
K(\mathbf{u},\mathbf{v})&=\alpha K_1(\mathbf{u},\mathbf{v}),\alpha>0,\\
\phi(\mathbf{x})&=\alpha\phi_1(\mathbf{x})\\
K(\mathbf{u},\mathbf{v})&=f(\mathbf{u})K_1(\mathbf{u},\mathbf{v})f(\mathbf{v}),\forall f:\mathcal{X}\rightarrow\mathbb{R},\\
\phi(\mathbf{x})&=f(\mathbf{x})\phi(\mathbf{x})\\
K(\mathbf{u},\mathbf{v})&=K_1(\mathbf{u},\mathbf{v})+K_2(\mathbf{u},\mathbf{v}),\phi(\mathbf{x})\in\mathbb{R}^{M_1+M_2},\\
\phi(\mathbf{x})&=
\begin{bmatrix*}[l]
\phi_1(\mathbf{x})^{(1)}\\
\;\;\;\;\;\;\vdots\\
\phi_1(\mathbf{x})^{(M_1)}\\
\phi_2(\mathbf{x})^{(1)}\\
\;\;\;\;\;\;\vdots\\
\phi_2(\mathbf{x})^{(M_2)}
\end{bmatrix*}\\
K(\mathbf{u},\mathbf{v})&=K_1(\mathbf{u},\mathbf{v})K_2(\mathbf{u},\mathbf{v}),\phi(\mathbf{x})\in\mathbb{R}^{M_1M_2},\\
\phi(\mathbf{x})&=
\begin{bmatrix*}[c]
\phi_1(\mathbf{x})^{(1)}\phi_2(\mathbf{x})^{(1)}\\
\phi_1(\mathbf{x})^{(1)}\phi_2(\mathbf{x})^{(2)}\\
\vdots\\
\phi_1(\mathbf{x})^{(M_1)}\phi_2(\mathbf{x})^{(M_2-1)}\\
\phi_1(\mathbf{x})^{(M_1)}\phi_2(\mathbf{x})^{(M_2)}
\end{bmatrix*}
\end{aligned}$$

Using these properties, we can come up with some useful kernel functions:

$$\begin{aligned}
K(\mathbf{u},\mathbf{v})&=\mathbf{u}\cdot\mathbf{v}&\text{ (Linear Kernel)}\\
K(\mathbf{u},\mathbf{v})&=(\mathbf{u}\cdot\mathbf{v}+1)^p&\text{ (Polynomial Kernel)}\\
K(\mathbf{u},\mathbf{v})&=e^{-\gamma\left\|\mathbf{u}-\mathbf{v}\right\|^2}&\text{ (RBF Kernel)}\\
\end{aligned}$$

While the linear and polynomial kernels may be obvious (use the addition and product rule), the RBF kernel can be hard to interpret:

$$\begin{aligned}
K(\mathbf{u},\mathbf{v})&=e^{-\gamma\left\|\mathbf{u}-\mathbf{v}\right\|^2}\\
&=e^{-\gamma\left\|\mathbf{u}\right\|^2-\gamma\left\|\mathbf{v}\right\|^2+2\gamma\mathbf{u}\cdot\mathbf{v}}\\
&=e^{-\gamma\left\|\mathbf{u}\right\|^2}e^{2\gamma\mathbf{u}\cdot\mathbf{v}}e^{-\gamma\left\|\mathbf{v}\right\|^2}
\end{aligned}$$

Now this looks like the second transformation above, we would like to prove the middle term a kernel. We will use Taylor expansion:

$$\begin{aligned}
K(\mathbf{u},\mathbf{v})&=e^{2\gamma\mathbf{u}\cdot\mathbf{v}}\\
&=\frac{(2\gamma\mathbf{u}\cdot\mathbf{v})^0}{0!}+\frac{(2\gamma\mathbf{u}\cdot\mathbf{v})^1}{1!}+...+\frac{(2\gamma\mathbf{u}\cdot\mathbf{v})^n}{n!}+...
\end{aligned}$$

So the middle term is in fact an infinite positive-weighted-sum of polynomial kernels, which is also a valid kernel. And we can tell that the feature mapping of a RBF kernel will have infinite dimensions, so it proves the importance of a kernel function as calculating the mapped features can be impossible.

### Sequential Minimal Optimization
*Reference*: [John C. Platt, Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines](https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf)

Now the only thing we need is to pick the multipliers to optimize the objective function. In another word, we are solving this **Quadratic Programming** problem:

$$\begin{aligned}
\max_{\textbf{α}}\;\;&{\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})}}}\\
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
\max_{\alpha_i,\alpha_j}\;\;&\alpha_i+\alpha_j-\frac{1}{2}\alpha_i^2K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\frac{1}{2}\alpha_j^2K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})\\
&-\alpha_iy^{(i)}\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\mathbf{x}^{(q)},\mathbf{x}^{(i)})}\\
&-\alpha_jy^{(j)}\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\mathbf{x}^{(q)},\mathbf{x}^{(j)})}\\
&-\alpha_i\alpha_jy^{(i)}y^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})+\delta\\
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
                   &&            &-\frac{1}{2}(\zeta-\alpha_j y^{(j)})^2K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\frac{1}{2}\alpha_j^2K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})\\
                   &&            &-(\zeta-\alpha_jy^{(j)})S_i-\alpha_jy^{(j)}S_j\\
                   &&            &-(\zeta-\alpha_j y^{(j)})\alpha_jy^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})+\delta\\
       \text{where}&&S_i=        &\;\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\mathbf{x}^{(q)},\mathbf{x}^{(i)})}\\
                   &&S_j=        &\;\sum_{1\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\mathbf{x}^{(q)},\mathbf{x}^{(j)})}
\end{aligned}
\end{aligned}
$$

To optimize, we take the partial derivative w/ respect to \\(\alpha_j\\):

$$\begin{aligned}
\frac{\partial J(\alpha_j)}{\partial\alpha_j}=&\;\alpha_j(2K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})-K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-K(\mathbf{x}^{(j)},\mathbf{x}^{(j)}))\\
&+\zeta y^{(j)}(K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}))\\
&+y^{(j)}(S_i-S_j)-y^{(i)}y^{(j)}+1\end{aligned}$$

If we look at the two sum terms \\(S_i\\), \\(S_j\\):

$$\begin{aligned}
S_i&=\sum_{0\leq q\leq n\atop q\neq i,j}{\alpha_qy^{(q)}K(\mathbf{x}^{(q)},\mathbf{x}^{(i)})}\\
&=\sum_{q=0}^n{\alpha_qy^{(q)}K(\mathbf{x}^{(q)},\mathbf{x}^{(i)})}-\alpha_iy^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\alpha_jy^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\\
&=\mathbf{w}\cdot\phi(\mathbf{x}^{(i)})-\alpha_iy^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\alpha_jy^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\\
&=f(\mathbf{x}^{(i)})-b-(\zeta-\alpha_jy^{(j)})K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\alpha_jy^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\\
S_j&=f(\mathbf{x}^{(j)})-b-\alpha_jy^{(j)}K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})-(\zeta-\alpha_jy^{(j)})K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})
\end{aligned}$$

We want to derive the optimized \\(\alpha_j\\) by making the derivative to 0, assume we are currently at step \\(k\\):

$$\begin{aligned}
y^{(j)}(S_i^k-S_j^k)=&\;y^{(j)}(f^{k}(\mathbf{x}^{(i)})-f^{k}(\mathbf{x}^{(j)}))-\zeta y^{(j)}(K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}))\\
&+\alpha_j^{k}(K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})+K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})-2K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}))\\
\frac{\partial J(\alpha_j)}{\partial\alpha_j}\big\rvert_{\alpha_j=\alpha_j^{k+1}}=&\;(\alpha_j^{k}-\alpha_j^{k+1})(K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})+K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})-2K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}))\\
&+y^{(j)}((f^{k}(\mathbf{x}^{(i)})-y^{(i)})-(f^{k}(\mathbf{x}^{(j)})-y^{(j)}))=0\\
\alpha_j^{k+1}=&\;\alpha_j^{k}+\frac{y^{(j)}(E_i^k-E_j^k)}{\eta}\\
\text{where }E_i^k,E_j^k&\text{ are the residuals, }\eta=K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})+K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})-2K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\end{aligned}$$

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
                  &\Rightarrow y^{(i)}(\mathbf{w}\cdot\phi(\mathbf{x}^{(i)})+b^{k+1})=1\\
                  &\;\begin{aligned}
                  \Rightarrow b_i^{k+1}=&\;y^{(i)}-(f^k(\mathbf{x}^{(i)})-b^{k}-\alpha_i^{k}y^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\alpha_j^{k}y^{(i)}K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})\\
                  &+\alpha_i^{k+1}y^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})-\alpha_j^{k+1}y^{(i)}K(\mathbf{x}^{(j)},\mathbf{x}^{(j)}))\\
                                     =&\;b^{k}-E_i^{k}+(\alpha_i^{k}-\alpha_i^{k+1})y^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(i)})+(\alpha_j^{k}-\alpha_j^{k+1})y^{(j)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})
                  \end{aligned}\\
&0<\alpha_j^{k+1}<C\\
&\Rightarrow\alpha_j^{k+1}\text{ on the margin }\\
                  &\Rightarrow b_j^{k+1}=b^{k}-E_j^{k}+(\alpha_j^{k}-\alpha_j^{k+1})y^{(j)}K(\mathbf{x}^{(j)},\mathbf{x}^{(j)})+(\alpha_i^{k}-\alpha_i^{k+1})y^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\\
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

