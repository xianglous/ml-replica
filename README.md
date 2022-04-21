# ml-replica
Replication of basic &amp; advanced ML models.
- [ml-replica](#ml-replica)
    - [Linear Clasiifiers](#linear-clasiifiers)
      - [Perceptron](#perceptron)

### Linear Clasiifiers
Linear classifiers classifies the input features based on the decision hyperplane in the feature space.

#### Perceptron 
The perceptron algorithm is the building block of deep learning. It updates on one data point at each time and moves in the right direction based on that point. <br>
*Pseudocode*
$$
\begin{quote}
k=0, W=0
\text{while not all correctly classified and k < max step}:\\
\tab\text{for }i\text{ in }1, ..., n:\\
\tab\tab\text{if }y_i(W\cdot X_i)\neq 0:
\tab\tab\tabW = W + yiXi
\tab\tab\tabk++
\end{quote}
$$
Mathematically, the perceptron update is
> $$\bar{\theta}^{(k+1)}=\bar{\theta}^{k}+y_i\bar{x}_i$$
