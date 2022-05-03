# ml-replica
Replication of basic &amp; advanced ML models.<br>

# Table of Contents
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
    - [Feature Mapping](#feature-mapping)
    - [Kernel Trick](#kernel-trick)
    - [Sequential Minimal Optimization](#sequential-minimal-optimization)

# Linear Clasiifiers
Linear classifiers classifies the input features based on the decision hyperplane in the feature space.

## Perceptron
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

## Stochastic Gradient Descent
Perceptron is nice and simple, but it has an important restriction: it only converges on linearly-separable data. <br>
To make it work for non-separable data, we need to change the way it approaches the best model. 

### Loss Functions
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

### Gradient Descent
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

### Stochastic Gradient Descent
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

## Support Vector Machine
As mentioned before, in linear classification problems we want to find a hyperplane that separates training data well. But there can be infinitely many hyperplanes that separate the data, we need to have additional measures to select the best ones. 

### Maximum Margin Separator
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

### Hard-Margin SVM
We can now formulate our problem as a constrained optimization. For computation purpose, we transform the maximization into a minimization problem:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\displaystyle\min_{\bar{w}}\;\;&{\frac{{\left\|\bar{w}\right\|}^2}{2}}\\\text{&space;subject&space;to&space;}\;&y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1,\forall&space;i\in\{1,...n\}\end{align}" />
</p>

### Lagrange Duality
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

The dual provides a lower bound for the primal solution, so there is a **duality gap** between the two formulations. The gap is 0 if the [**Karush–Kuhn–Tucker**](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) (**KKT**) conditions are satisfied:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}\nabla_{\bar{\theta}}L(\bar{\theta},\bar{\lambda})&=\mathbf{0}\\\nabla_{\bar{\lambda}}L(\bar{\theta},\bar{\lambda})&=\mathbf{0}\\\lambda_ih_i(\bar{\theta})&=0\\h_i(\bar{\theta})&\leq0\\\lambda_i&\geq0\end{align}" />
</p>

For our hard-margin SVM, the gap is 0. The Lagrangian function is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}L(\bar{w},b,\bar{\alpha})=\frac{\left\|\bar{w}\right\|^2}{2}&plus;\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b))}" />
</p>

To satisfy the **KKT** conditions, we need the gradient with respect to `w` and `b` to be 0:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\displaystyle\nabla_{\bar{w}}L(\bar{w},b,\bar{\alpha})=\bar{w}-\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}=\mathbf{0}&\Rightarrow\bar{w}^*=\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}}\\\nabla_{b}L(\bar{w},b,\bar{\alpha})=-\sum_{i=1}^n{\alpha_iy^{(i)}}=0&\Rightarrow\sum_{i=1}^n{\alpha_iy^{(i)}}=0\end{align}"/>
</p>

Using the dual formation, our problem become:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b}L(\bar{w},b,\bar{\alpha})&=\max_{\bar{\alpha},\alpha_i\geq0}\min_{\bar{w},b}\frac{\left\|\bar{w}\right\|^2}{2}&plus;\sum_{i=1}^n{\alpha_i(1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}&plus;b))}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}(\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)})}\cdot(\sum_{i=1}^n{\alpha_iy^{(i)}\bar{x}^{(i)}})&plus;\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\alpha_iy^{(i)}\sum_{j=1}^n{\alpha_jy^{(j)}\bar{x}^{(j)}}\cdot\bar{x}^{(i)}}-b\sum_{i=1}^n{\alpha_iy^{(i)}}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}&plus;\sum_{i=1}^n{\alpha_i}-\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\\&=\max_{\bar{\alpha},\alpha_i\geq0}\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\bar{x}^{(i)}}\cdot\bar{x}^{(j)}}\end{align}"/>
</p>

According to the **complementary slackness** condition <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\alpha^*_i(1-y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*))=0" />:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}&\alpha^*_i>0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*)=1&\text{&space;(support&space;vector)}\\&\alpha^*_i=0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*)>1&\text{&space;(non-support&space;vector)}\end{align}&space;" />
</p>

We can also compute the intercept `b` using the support vectors:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}&\forall\alpha_k>0,y^{(k)}(\bar{w}^*\cdot\bar{x}^{(k)}&plus;b^*)=1\Rightarrow\bar{w}^*\cdot\bar{x}^{(k)}&plus;b^*=y^{(k)}\\&\Rightarrow&space;b^*=y^{(k)}-\bar{w}^*\cdot\bar{x}^{(k)}\end{align}&space;" />
</p>

### Soft-Margin SVM
However the hard-margin SVM above has limitations. If the data is not linearly separable, the SVM algorithm may not work. Consider the following example:

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/soft-margin.png" width=300/>
</p>

If we use hard-margin SVM, the fitted model will be highly affected by the single outlier red point. But if we allow some misclassification by adding in the **slack variables**, the final model may be more robust. The setup for a soft-margin SVM is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\displaystyle\min_{\bar{w},b,\bar{\xi}}\;\;&{\frac{{\left\|\bar{w}\right\|}^2}{2}&plus;C\sum_{i=1}^n{\xi_i}},\\\text{&space;subject&space;to&space;}\;&\xi_i\geq&space;0,y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}+b)\geq1-\xi_i,\forall&space;i\in\{1,...n\}\end{align}" />
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

We can see that soft-margin SVM has a same dual formulation as the hard-margin SVM. And now, the condition for optimum are <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\alpha^*_i(1-\xi^*_iy^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}+b^*))=0" /> **AND** <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\beta^*_i(-\xi^*_i)=0" />, so combining them together:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}&\alpha^*_i=0\Rightarrow\beta^*_i=C\Rightarrow\xi^*_i=0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}&plus;b^*)\geq1-\xi^*_i=1&\text{&space;(non-support&space;vector)}\\&\alpha^*_i=C\Rightarrow\beta^*_i=0\Rightarrow\xi^*_i\geq0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}&plus;b^*)=1-\xi_i^*\leq1&\text{&space;(support&space;vector&space;off&space;the&space;margin)}\\&0<\alpha_i^*<C\Rightarrow0<\beta^*_i<C\Rightarrow\xi^*_i=0\Rightarrow&space;y^{(i)}(\bar{w}^*\cdot\bar{x}^{(i)}&plus;b^*)=1-\xi^*_i=1&\text{&space;(support&space;vector&space;on&space;the&space;margin)}\end{align}" />
</p>

An observation that can be drawn from this result is that `C` is a hyperparameter that controls the "softness" of our SVM model. If `C` is big enough, the soft-margin SVM will become a hard-margin one.

### Feature Mapping
Soft-margin SVM seems to provide a decent approach to non-linearly separable data, but it only works well when there are a few "noisy" data. When the boundary between categories is inherently non-linear, it is not reasonable to use a soft-margin SVM. For [example](http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html):

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/non-sep.png" width=300/>
</p>

We cannot find a separating hyperplane, so the SVMs we have discussed will not work as expected. However, if we jump out of the 2-dimensional space, we can find hyperplanes that can separate the data:

<p align="center">
<img src="https://github.com/xianglous/ml-replica/blob/main/Illustration/feature-map.png" width=300/>
</p>

The way we convert the lower-dimensional coordinates into higher ones is called a **feature mapping**.

### Kernel Trick
Assume we have a feature mapping <img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}\phi(\cdot):\mathcal{X}\rightarrow\mathcal{F}" /> and we fit this mapped data using SVM, then the objective function would be:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}J(\bar{\alpha})=\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}\left<\phi(\bar{x}^{(i)}),\phi(\bar{x}^{(j)})\right>}}\end{align}" />
</p>

One problem with this setup is that the computation can be slow because we need to 1) map the features to higher dimension, 2) compute the inner products between each pair of mapped features. Also, note that predicting the response for a new data point is:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}\hat{y}&=\text{sgn}((\bar{w}^*)^\top\phi(\hat{\bar{x}})&plus;b^*)=\text{sgn}(\sum_{i=1}^n{\alpha_iy^{(i)}\left<\phi(\bar{x}^{(i)}),\phi(\hat{\bar{x}})\right>}&plus;b^*)\\b^*&=y^{(k)}-(\bar{w}^*)^\top\phi(\bar{x}^{(k)})=y^{(k)}-\sum_{i=1}^n{\alpha_iy^{(i)}\left<\phi(\bar{x}^{(i)}),\phi(\bar{x}^{(k)})\right>},\forall\alpha_k>0\end{align}" />
</p>

We can see that only the inner product of the mappings are needed in training or evaluation. So instead of computing the mapping, we would like to compute the inner products of the mapped features directly. Therefore, we introduce the **kernel function**:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}&K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}\\&\text{&space;s.t.&space;}\exists\phi:\mathcal{X}\rightarrow\mathcal{F},\forall&space;\bar{x},\bar{x}'\in\mathcal{X}\Rightarrow&space;K(\bar{x},\bar{x}')=\left<\phi(\bar{x}),\phi(\bar{x}')\right>\end{align}" />
</p>

Then we can rewrite the objective function as:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}J(\bar{\alpha})=\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})}}\end{align}" />
</p>

According to [**Mercer's Theorem**](https://xavierbourretsicotte.github.io/Kernel_feature_map.html#Necessary-and-sufficient-conditions), a kernel function is valid if and only if its **Gram matrix** must be positive semi-definite. Below are some properties of kernel functions, let <img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}K_1,K_2:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}"> be two valid kernels of feature mapping <img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\phi_1:\mathcal{X}\rightarrow\mathbb{R}^{M_1},\phi_2:\mathcal{X}\rightarrow\mathbb{R}^{M_2}">, then the following kernels and feature maps <img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R},\phi:\mathcal{X}\rightarrow\mathcal{F}"> are valid:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\begin{align*}K(\bar{u},\bar{v})&=\alpha&space;K_1(\bar{u},\bar{v}),\alpha>0,\\\phi(\bar{x})&=\alpha\phi_1(\bar{x})\\K(\bar{u},\bar{v})&=f(\bar{u})K_1(\bar{u},\bar{v})f(\bar{v}),\forall&space;f:\mathcal{X}\rightarrow\mathbb{R},\\\phi(\bar{x})&=f(\bar{x})\phi(\bar{x})\\K(\bar{u},\bar{v})&=K_1(\bar{u},\bar{v})&plus;K_2(\bar{u},\bar{v}),\phi(\bar{x})\in\mathbb{R}^{M_1&plus;M_2},\\\phi(\bar{x})&=\[\phi_1(\bar{x})^{(1)},...,\phi_1(\bar{x})^{(M_1)},\phi_2(\bar{x})^{(1)},...,\phi_2(\bar{x})^{(M_2)}\]^\top\\K(\bar{u},\bar{v})&=K_1(\bar{u},\bar{v})K_2(\bar{u},\bar{v}),\phi(\bar{x})\in\mathbb{R}^{M_1M_2},\\\phi(\bar{x})&=\[\phi_1(\bar{x})^{(1)}\phi_2(\bar{x})^{(1)},\phi_1(\bar{x})^{(1)}\phi_2(\bar{x})^{(2)},...,\phi_1(\bar{x})^{(M_1)}\phi_2(\bar{x})^{(M_2-1)},\phi_1(\bar{x})^{(M_1)}\phi_2(\bar{x})^{(M_2)}\]^\top\\\end{align}" />
</p>

Using these properties, we can come up with some useful kernel functions:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}K(\bar{u},\bar{v})&=\bar{u}\cdot\bar{v}&\text{&space;(Linear&space;Kernel)}\\K(\bar{u},\bar{v})&=(\bar{u}\cdot\bar{v}&plus;1)^p&\text{&space;(Polynomial&space;Kernel)}\\K(\bar{u},\bar{v})&=e^{-\gamma\left\|\bar{u}-\bar{v}\right\|^2}&\text{&space;(RBF&space;Kernel)}\\\end{align}" />
</p>

While the linear and polynomial kernels may be obvious (use the addition and product rule), the RBF kernel can be hard to interpret:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}K(\bar{u},\bar{v})&=e^{-\gamma\left\|\bar{u}-\bar{v}\right\|^2}\\&space;&=e^{-\gamma\left\|\bar{u}\right\|^2-\gamma\left\|\bar{v}\right\|^2+2\gamma\bar{u}\cdot\bar{v}}\\&=e^{-\gamma\left\|\bar{u}\right\|^2}e^{2\gamma\bar{u}\cdot\bar{v}}e^{-\gamma\left\|\bar{v}\right\|^2}\end{align}" />
</p>

Now this looks like the second property above, we would like to prove the middle term a kernel. We will use Taylor expansion:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{100}\bg{white}\begin{align*}K(\bar{u},\bar{v})&=e^{2\gamma\bar{u}\cdot\bar{v}}\\&=\frac{(2\gamma\bar{u}\cdot\bar{v})^0}{0!}&plus;\frac{(2\gamma\bar{u}\cdot\bar{v})^1}{1!}&plus;...&plus;\frac{(2\gamma\bar{u}\cdot\bar{v})^n}{n!}&plus;...\end{align}" />
</p>

So the middle term is in fact a infinite sum of scalar-multiplied polynomial kernels, which is also a valid kernel. And we can tell that the feature mapping of a RBF kernel will have infinite dimensions, so it proves the importance of a kernel function as calculating the mapped feature can be impossible.

### Sequential Minimal Optimization
*Reference*: https://cs229.stanford.edu/lectures-spring2022/master.pdf

Now the only thing we need is to pick the multipliers to optimize the objective function. In another word, we are solving this **Quadratic Programming** problem:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\max_{\bar{\alpha}}\;\;&{\sum_{i=1}^n{\alpha_i}-\frac{1}{2}\sum_{i=1}^n{\sum_{j=1}^n{\alpha_i\alpha_jy^{(i)}y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})}}}\\\text{subject&space;to&space;}\;&0\leq\alpha_i\leq&space;C,&space;\forall&space;i&space;=&space;1...n,\\&\sum_{i=1}^n{\alpha_i&space;y^{(i)}}=0\end{align}" />
</p>

The main idea of the **Sequential Minimal Optimization (SMO)** algorithm is to optimize only a **pair** of multipliers each time. It works as following:

*Pseudocode*:
<pre>
<b>α</b>=0, b=0 
<b>while</b> not all α satisfies <b>KKT</b> conditions:
    <b>pick</b> αi, αj using some <b>heuristics</b>
    <b>optimize</b> αi, αj
    <b>update</b> b
</pre>

The optimization for each pair can be represented as:
<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\max_{\alpha_i,\alpha_j}\;\;&\alpha_i&plus;\alpha_j-\frac{1}{2}\alpha_i^2K(\bar{x}^{(i)},\bar{x}^{(i)})-\frac{1}{2}\alpha_j^2K(\bar{x}^{(j)},\bar{x}^{(j)})-\alpha_i\alpha_jy^{(i)}y^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\&-\alpha_iy^{(i)}\sum_{\substack{1\leq&space;q\leq&space;n\\q\neq&space;i,j}}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}-\alpha_jy^{(j)}\sum_{\substack{1\leq&space;q\leq&space;n\\q\neq&space;i,j}}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(j)})}-\delta\\\text{subject&space;to}\;\;&0\leq\alpha_i,\alpha_j\leq&space;C\\&\alpha_iy^{(i)}&plus;\alpha_jy^{(j)}=-\sum_{\substack{1\leq&space;q\leq&space;n\\q\neq&space;i,j}}\alpha_qy^{(q)}=\zeta\end{align}" />
</p>

Now we can substitute αj for αi:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\alpha_iy^{(i)}&plus;\alpha_jy^{(j)}=&\zeta\Rightarrow\alpha_i=\zeta&space;y^{(i)}-\alpha_j&space;y^{(i)}y^{(j)}\\\Rightarrow&space;J(\alpha_j)=&\zeta&space;y^{(i)}-\alpha_j&space;y^{(i)}y^{(j)}&plus;\alpha_j-\frac{1}{2}(\zeta-\alpha_j&space;y^{(j)})^2K(\bar{x}^{(i)},\bar{x}^{(i)})-\frac{1}{2}\alpha_j^2K(\bar{x}^{(j)},\bar{x}^{(j)})\\&-(\zeta-\alpha_j&space;y^{(j)})\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})-(\zeta-\alpha_jy^{(j)})S_i-\alpha_jy^{(j)}S_j-\delta\\\text{where&space;}S_i=&\sum_{\substack{1\leq&space;q\leq&space;n\\q\neq&space;i,j}}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})},S_j=\sum_{\substack{1\leq&space;q\leq&space;n\\q\neq&space;i,j}}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(j)})}\end{align}" />
</p>

To optimize, we take the partial derivative w/ respect αj:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\frac{\partial&space;J(\alpha_j)}{\partial\alpha_j}=&\alpha_j(2K(\bar{x}^{(i)},\bar{x}^{(j)})-K(\bar{x}^{(i)},\bar{x}^{(i)})-K(\bar{x}^{(j)},\bar{x}^{(j)}))\\&&plus;\zeta&space;y^{(j)}(K(\bar{x}^{(i)},\bar{x}^{(i)})-K(\bar{x}^{(i)},\bar{x}^{(j)}))\\&&plus;y^{(j)}(S_i-S_j)-y^{(i)}y^{(j)}&plus;1\end{align}" />
</p>

If we look at the two sum terms Si, Sj:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}S_i&=\sum_{\substack{0\leq&space;q\leq&space;n\\q\neq&space;i,j}}{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}=\sum_{q=0}^n{\alpha_qy^{(q)}K(\bar{x}^{(q)},\bar{x}^{(i)})}-\alpha_iy^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\&=\bar{w}\cdot\phi(\bar{x}^{(i)})-\alpha_iy^{(i)}K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\&=f(\bar{x}^{(i)})-b-(\zeta-\alpha_jy^{(j)})K(\bar{x}^{(i)},\bar{x}^{(i)})-\alpha_jy^{(j)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\S_j&=\bar{w}\cdot\phi(\bar{x}^{(j)})-\alpha_jy^{(j)}K(\bar{x}^{(j)},\bar{x}^{(j)})-\alpha_iy^{(i)}K(\bar{x}^{(i)},\bar{x}^{(j)})\\&=f(\bar{x}^{(j)})-b-\alpha_jy^{(j)}K(\bar{x}^{(j)},\bar{x}^{(j)})-(\zeta-\alpha_jy^{(j)})K(\bar{x}^{(i)},\bar{x}^{(j)})\\\end{align}" />
</p>

We want to derive the optimized αj by making the derivative to 0, assume we are currently at step k:

<p align="center">
<img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}\begin{align*}\Rightarrow&space;y^{(j)}(S_i^k-S_j^k)=&y^{(j)}(f^{k}(\bar{x}^{(i)})-f^{k}(\bar{x}^{(j)}))-\zeta&space;y^{(j)}(K(\bar{x}^{(i)},\bar{x}^{(i)})-K(\bar{x}^{(i)},\bar{x}^{(j)}))\\&&plus;\alpha_j^{k}(K(\bar{x}^{(i)},\bar{x}^{(i)})&plus;K(\bar{x}^{(j)},\bar{x}^{(j)})-2K(\bar{x}^{(i)},\bar{x}^{(j)}))\\\Rightarrow\frac{\partial&space;J(\alpha_j)}{\partial\alpha_j}\big\rvert_{\alpha_j=\alpha_j^{k&plus;1}}=&(\alpha_j^{k}-\alpha_j^{k&plus;1})(K(\bar{x}^{(i)},\bar{x}^{(i)})&plus;K(\bar{x}^{(j)},\bar{x}^{(j)})-2K(\bar{x}^{(i)},\bar{x}^{(j)}))\\&&plus;y^{(j)}((f^{k}(\bar{x}^{(i)})-y^{(i)})-(f^{k}(\bar{x}^{(j)})-y^{(j)}))=0\\\Rightarrow\alpha_j^{k&plus;1}=&\alpha_j^{k}&plus;\frac{y^{(j)}(E_i^k-E_j^k)}{\eta}\text{,&space;where&space;}E_i^k,E_j^k\text{&space;are&space;the&space;residuals,}\\\eta=&K(\bar{x}^{(i)},\bar{x}^{(i)})&plus;K(\bar{x}^{(j)},\bar{x}^{(j)})-2K(\bar{x}^{(i)},\bar{x}^{(j)})\end{align}" />
</p>

Therefore, we are able to use the residual values and the kernel function to calculate the optimized αj, and thus αi and b:
