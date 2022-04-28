# ml-replica
Replication of basic &amp; advanced ML models.<br>

## Table of Contents
- [Linear Clasiifiers](#linear-clasiifiers)
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

### Linear Clasiifiers
Linear classifiers classifies the input features based on the decision hyperplane in the feature space.

#### Perceptron
The perceptron algorithm is the building block of deep learning. It updates on one data point at each time and moves in the "right" direction based on that point. <br><br>
*Pseudocode* (w/o offset)
<pre>
k=0, w=0
<b>while</b> not all correctly classified <b>and</b> k < max step:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w.Xi) <= 0: // misclassified
            w = w + yiXi // the Update
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

*Code*: [perceptron.py](https://github.com/xianglous/ml-replica/blob/main/Linear%20Classifiers/perceptron.py)

### Stochastic Gradient Descent
Perceptron is nice and simple, but it has an important restriction: it only converges on linearly-separable data. <br>
To make it work for non-separable data, we need to change the way it approaches the best model. 

#### Loss Functions
In machine learning, we often use a loss function to measure the fit of the current model to the training data. For example, the perceptron algorithm uses the following loss function:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}L(X,&space;\bar{y},&space;\bar{w})=\frac{1}{n}\sum_{i=1}^n{\[y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})\leq&space;;0]}"/>
</p>
A problem with this loss function is that it does not measures the distance between the predicted and actual value, so 0.1 and 1 will all be seen as a good classification while -0.1 and -1 will all be equally bad. <br>

So another loss function we can use instead is the **Hinge Loss**, for each fitted value, the Hingle Loss is:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}h(\bar{x}^{(i)},&space;y^{(i)},&space;\bar{w})=\max(0,&space;1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}))"/>
</p>
And for the whole model, the loss function is defined as:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}L(X,&space;\bar{y},&space;\bar{w})=\frac{1}{n}\sum_{i=1}^n{\max(0,&space;1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})))}"/>
</p>
This loss will penalize any imperfect prediction.

#### Gradient Descent
The loss function tells us about how **bad** the current model fits the data. Therefore, we need to know the direction in which moving the parameters will decrease the loss. In mathematics, we use the gradient to measure the "direction." For Hinge Loss, the gradient of a single data point is: 
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\nabla_{\bar{w}}{h(\bar{x}^{(i)},&space;y^{(i)},\bar{w})}=\left\{\begin{matrix}-y^{(i)}\bar{x}^{(i)}&\text{if&space;}y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})<1\\\mathbf{0}&space;&&space;\text{otherwise}\end{matrix}\right."/>
</p>
And the gradient of the whole training data is:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\nabla_{\bar{w}}{L(X,&space;\bar{y},\bar{w})}=\frac{1}{n}\sum_{i=1}^n{\nabla_{\bar{w}}{h(\bar{x}^{(i)},&space;y^{(i)},\bar{w})}}"/>
</p>

By moving the the weights in the direction of the gradient, we will likely decrease the loss of our model. So the **Gradient Descent** is defined as:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\bar{w}^{(k&plus;1)}=\bar{w}^{(k)}-\eta\nabla_{\bar{w}}{L(X,&space;\bar{y},&space;\bar{w})}"/>
</p>

*Pseudocode*
<pre>
k=0, w=0
<b>while</b> criterion not met:
    g = 0
    <b>for</b> i in 1...n:
        <b>if</b> yi(w.Xi)<1:
            g = g - yiXi/n
    w = w - η*g // the Descent
    k++
</pre>
η is the step size, or the learning rate.

#### Stochastic Gradient Descent
The problem with gradient descent is that we need to compute the gradient of each data point in every iteration, which can be slow when the training data is huge. Alternatively, we can update based on a single data point in each iteration, and that is **Stochastic Gradient Descent**.<br>

*Pseudocode*
<pre>
k=0, w=0
<b>while</b> criterion not met:
    <b>for</b> i in 1...n:
        <b>if</b> yi(w.Xi)<1:
            w = w + η*yiXi // the Descent
            k++
</pre>

*Code*: [sgd.py](https://github.com/xianglous/ml-replica/blob/main/Linear%20Classifiers/sgd.py)

### Support Vector Machine
As mentioned before, in linear classification problems we want to find a hyperplane that separates training data well. But there can be infinitely many hyperplanes that separate the data, we need to have additional measures to select the best ones. 

#### Maximum Margin Separator
Consider the following example:

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/non_max_margin.png" width=300/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/max_margin.png" width=300/>
</p>

The solid lines on each figure represents a classification boundary that separates the two training classes. They both perfectly classify the training data, but which one is better? We would consider the right one better because it maximizes the "margins" between the separator and the training data.<br>
So while using the Hinge Loss may produce either of the above models, maximizing the margin will give us the better model. In another word, we want to maximize the distance from the closest point to the separating hyperplane:

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/max_margin_distance.png" width=300/>
</p>

If we look at the two margine lines, they are actually the decision lines <img src="https://latex.codecogs.com/png.image?\bg{white}\inline&space;\dpi{110}\bar{w}\cdot\bar{x}&plus;b=1" /> and <img src="https://latex.codecogs.com/png.image?\bg{white}\inline&space;\dpi{110}\bar{w}\cdot\bar{x}&plus;b=-1" /> beecause they are right on the border of being penalized. So we can calculate the margin as the distance between the positive margin line and the decision boundary:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}d=\frac{|(1-b)-(-b)|}{\left\|\bar{w}\right\|}=\frac{1}{\left\|\bar{w}\right\|}"/>
</p>

And we want our model to maximize the margin:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\displaystyle\max_{\bar{w},&space;b}{\frac{1}{\left\|\bar{w}\right\|}}" />
</p>

#### Hard-Margin SVM
We can now formulate our problem as a constrained optimization. For computation purpose, we transform the maximization into a minimization problem:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}&\displaystyle\min_{\bar{w},b}{\frac{{\left\|\bar{w}\right\|}^2}{2}},\\&\text{&space;subject&space;to&space;}y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1,\forall&space;i\in\{1,...n\}\end{align}" />
</p>

#### Lagrange Duality
For a constrained optimization problem <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\min_{\bar{\theta}}{f(\bar{\theta})}" /> subject to n constraints <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}h_i(\bar{\theta})\leq0,\forall&space;i\in\{1,...,n\}" />, we can combine the objective function with the contraints using the **Lagrange multipliers** <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\lambda_1,...,\lambda_n\geq0" />. 

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}L(\bar{\theta},\bar{\lambda})=f(\theta)+\sum_{i=1}^n{\lambda_ih_i(\bar{\theta})}" />
</p>

From this formation, we observe that if a model satifies all the constraints, <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}f(\bar{\theta})\geq&space;L(\bar{\theta},\bar{\lambda})" />, so minimizing `f` is the same as minimizing the maximum of `L`, that is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\displaystyle\min_{\bar{\theta}}\max_{\bar{\lambda},\lambda_i\geq0}{L(\bar{\theta},\bar{\lambda}})" />
</p>

For our hard-margin SVM, the Lagrangian function is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}L(X,\bar{y},\bar{w},\bar{\lambda})=\frac{\left\|\bar{w}\right\|^2}{2}&plus;\sum_{i=1}^n{\lambda_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}&plus;b))}" />
</p>

To minimize the objective, we want the gradient of the Lagrangian with respect to `λ` and `w` be zero. So,


#### Soft-Margin SVM
However the hard-margin SVM above has limitations. If the data is not linearly separable, the SVM algorithm may not work. Consider the following example:

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/soft-margin.png" width=300/>
</p>

If we use hard-margin SVM, the fitted model will be highly affected by the single outlier red point. But if we allow some misclassification, i.e. a soft-margine, the final model may be more robust. The setup for a soft-margin SVM is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}&\displaystyle\min_{\bar{w},b,\bar{\xi}}{\frac{{\left\|\bar{w}\right\|}^2}{2}&plus;C\sum_{i=1}^n{\xi_i}},\\&\text{&space;subject&space;to&space;}\xi_i\geq&space;0,y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1-\xi_i,\forall&space;i\in\{1,...n\}\end{align}" />
</p>
