<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Linear Clasiifiers](#linear-clasiifiers)
  - [Perceptron](#perceptron)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
    - [Loss Functions](#loss-functions)
    - [Gradient Descent](#gradient-descent)

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

$$L(X, \bar{y}, \bar{w})=\frac{1}{n}\sum_{i=1}^n{[y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})\leq ;0]}$$

A problem with this loss function is that it does not measures the distance between the predicted and actual value, so 0.1 and 1 will all be seen as a good classification while -0.1 and -1 will all be equally bad. <br>

So another loss function we can use instead is the **Hinge Loss**, for each fitted value, the Hingle Loss is:

$$h(\bar{x}^{(i)}, y^{(i)}, \bar{w})=\max(0, 1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)}))$$

And for the whole model, the loss function is defined as:

$$L(X, \bar{y}, \bar{w})=\frac{1}{n}\sum_{i=1}^n{\max(0, 1-y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})))}$$

This loss will penalize any imperfect prediction.

### Gradient Descent
The loss function tells us about how **bad** the current model fits the data. Therefore, we need to know the direction in which moving the parameters will decrease the loss. In mathematics, we use the gradient to measure the "direction." For Hinge Loss, the gradient of a single data point is: 

$$\nabla_{\bar{w}}{h(\bar{x}^{(i)}, y^{(i)},\bar{w})}=\left\{\begin{matrix}-y^{(i)}\bar{x}^{(i)}&\text{if }y^{(i)}(\bar{w}\cdot\bar{x}^{(i)})<1\\\mathbf{0} & \text{otherwise}\end{matrix}\right$$

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
        <b>if</b> yi(w.Xi)<1:
            g = g - yiXi/n
    w = w - η*g // the Descent
    k++
</pre>
η is the step size, or the learning rate.
