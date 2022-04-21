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
`
k=0, w=0<br>
while not all correctly classified and k < max step:
    for i in 1...n:
        if yi(w•Xi) <= 0: // misclassified
            w = w + yi\*Xi // the Update
            k++
`


