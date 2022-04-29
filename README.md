# ml-replica
Replication of basic &amp; advanced ML models.<br>

## Table of Contents
- [ml-replica](#ml-replica)
  - [Table of Contents](#table-of-contents)
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
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}&\displaystyle\min_{\bar{w}}{\frac{{\left\|\bar{w}\right\|}^2}{2}},\\&\text{&space;subject&space;to&space;}y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})\geq1,\forall&space;i\in\{1,...n\}\end{align}" />
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

This is called the **primal formulation**. And we have **dual formulation**:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\displaystyle\max_{\bar{\lambda},\lambda_i\geq0}\min_{\bar{\theta}}{L(\bar{\theta},\bar{\lambda}})" />
</p>

The dual provides a lower bound for the primal solution, so there is a **duality gap** between the two formulations. Under certain conditions (strong duality), the gap is 0.

For our hard-margin SVM, the gap is 0. The Lagrangian function is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}L(\bar{w},b,\bar{\alpha})=\frac{\left\|\bar{w}\right\|^2}{2}&plus;\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b))}" />
</p>

To optimize, we need the gradient with respect to `w` and `b` to be 0:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\displaystyle\nabla_{\bar{w}}L(\bar{w},b,\bar{\alpha})=\bar{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}=\mathbf{0}&\Rightarrow\bar{w}^*=\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}\\\nabla_{b}L(\bar{w},b,\bar{\alpha})=-\sum_{i=1}^n{\alpha_iy^{(i)}}=0&\Rightarrow\sum_{i=1}^n{\alpha_iy^{(i)}}=0\end{align}"/>
</p>

Using the dual formation, our problem become:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b}L(\bar{w},b,\bar{\alpha})&=\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b}\frac{\left\|\bar{w}\right\|^2}{2}&plus;\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}&plus;b))}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}(\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)})}\cdot(\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}})&plus;\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\alpha_iy^{(i)}\sum_{j=1}^n{\alpha_jy^{(j)}\bar{x}^{(j)}}\cdot\bar{x}^{(i)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}&plus;\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\end{align}"/>
</p>

According to the **complementary slackness** condition for optimum in Lagrange duality problem <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\alpha^*_i(1-y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*))=0" />:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}&\alpha^*_i>0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*)=1&\text{&space;(support&space;vector)}\\&\alpha^*_i=0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*)>1&\text{&space;(non-support&space;vector)}\end{align}&space;" />
</p>

We can also compute the intercept `b` using the support vectors:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}&\forall\alpha_k>0,y^{(k)}(\bar{w}^*\cdot\bar{x}^{(k)}&plus;b^*)=1\Rightarrow\bar{w}^*\cdot\bar{x}^{(k)}&plus;b^*=y^{(k)}\\&\Rightarrow&space;b^*=y^{(k)}-\bar{w}^*\cdot\bar{x}^{(k)}\end{align}&space;" />
</p>

#### Soft-Margin SVM
However the hard-margin SVM above has limitations. If the data is not linearly separable, the SVM algorithm may not work. Consider the following example:

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/soft-margin.png" width=300/>
</p>

If we use hard-margin SVM, the fitted model will be highly affected by the single outlier red point. But if we allow some misclassification by adding in the **slack variables**, the final model may be more robust. The setup for a soft-margin SVM is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}&\displaystyle\min_{\bar{w},b,\bar{\xi}}{\frac{{\left\|\bar{w}\right\|}^2}{2}&plus;C\sum_{i=1}^n{\xi_i}},\\&\text{&space;subject&space;to&space;}\xi_i\geq&space;0,y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1-\xi_i,\forall&space;i\in\{1,...n\}\end{align}" />
</p>

The Lagrangian is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}\displaystyle&space;L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=\frac{\left\|\bar{w}\right\|^2}{2}&plus;C\sum_{i=1}^n{\xi_i}&plus;\sum_{i=1}^n{\alpha_i(1-\xi_i-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}&plus;b))}&plus;\sum_{i=1}^n{\beta_i(-\xi_i)}\end{align}" />
</p>

We first find the gradient with respect to `w` `b`, and the slack vector:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}\nabla_{\bar{w}}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=\bar{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}=\mathbf{0}&\Rightarrow\bar{w}^*=\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}\\\nabla_{b}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=-\sum_{i=1}^n{\alpha_iy^{(i)}}=0&\Rightarrow\sum_{i=1}^n{\alpha_iy^{(i)}}=0\\\nabla_{\bar{\xi}}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})=\begin{bmatrix}&space;C-\alpha_1-\beta_1\\&space;\vdots\\C-\alpha_n-\beta_n\end{bmatrix}=\mathbf{0}&\Rightarrow\alpha_i=C-\beta_i\Rightarrow0\leq\alpha_i\leq&space;C\end{align}&space;" />
</p>

And the dual formulation is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b,\bar{\xi}}L(\bar{w},b,\bar{\xi},\bar{\alpha},\bar{\beta})&=\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b,\bar{\xi}}\frac{\left\|\bar{w}\right\|^2}{2}&plus;C\sum_{i=1}^n{\xi_i}&plus;\sum_{i=1}^n{\alpha_i(1-\xi_i-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}&plus;b))}&plus;\sum_{i=1}^n{\beta_i(-\xi_i)}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}&plus;\sum_{i=1}^n{(C-\alpha_i-\beta_i)\xi_i}&plus;\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\end{align}" />
</p>

We can see that soft-margin SVM has a same dual formulation as the hard-margin SVM. And now, the condition for optimum is <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\alpha^*_i(1-\xi^*_iy^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*))=0" /> **AND** <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\beta^*_i(-\xi^*_i)=0" />, so combining them together:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}&\alpha^*_i=0\Rightarrow\beta^*_i=C\Rightarrow\xi^*_i=0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}&plus;b^*)\geq1-\xi^*_i=1&\text{&space;(non-support&space;vector)}\\&\alpha^*_i=C\Rightarrow\beta^*_i=0\Rightarrow\xi^*_i\geq0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}&plus;b^*)=1-\xi_i^*\leq1&\text{&space;(support&space;vector&space;off&space;the&space;margin)}\\&0<\alpha_i^*<C\Rightarrow0<\beta^*_i<C\Rightarrow\xi^*_i=0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}&plus;b^*)=1-\xi^*_i=1&\text{&space;(support&space;vector&space;on&space;the&space;margin)}\end{align}" />
</p>
