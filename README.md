# ml-replica
Replication of basic &amp; advanced ML models.
- [ml-replica](#ml-replica)
    - [Linear Clasiifiers](#linear-clasiifiers)
      - [Perceptron](#perceptron)

### Linear Clasiifiers
Linear classifiers classifies the input features based on the decision hyperplane in the feature space.

#### Perceptron 
The perceptron algorithm is the building block of deep learning. It updates on one data point at each time and moves in the right direction based on that point. <br>
*Pseudocode* <br>
> k=0, w=0<br>
> while not all correctly classified and k < max step:<br>
> &nbsp;for i in 1...n:<br>
> &nbsp;&nbsp;if yi(w•Xi) <= 0: // misclassified<br>
> &nbsp;&nbsp;&nbsp;w = w + yi\*Xi // the Update<br>
> &nbsp;&nbsp;&nbsp;k++<br>


