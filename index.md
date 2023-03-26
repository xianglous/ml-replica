---
layout: home
title: ML-Replica
---

{% include head.html %}

# Introduction
**ML-Replica** is a repository that contains implementations of basic and adavanced machine learning models and algorithms. It has a structure inspired by the [**Scikit-Learn**](https://github.com/scikit-learn/scikit-learn) library. The model definition and implementation are inside the [mlreplica](https://github.com/xianglous/ml-replica/tree/main/mlreplica) folder, and the example usage are inside the [test](https://github.com/xianglous/ml-replica/tree/main/tests) folder. Just like sklearn, models are classified into categories including [linear_model](https://github.com/xianglous/ml-replica/tree/main/mlreplica/linear_model), [tree_model](https://github.com/xianglous/ml-replica/tree/main/mlreplica/tree_model), [ensemble_model](https://github.com/xianglous/ml-replica/tree/main/mlreplica/ensemble_model), and [to-be-implemented](#repository-structure). There is also a [utils](https://github.com/xianglous/ml-replica/tree/main/mlreplica/utils) module that contains implementation of [datasets](https://github.com/xianglous/ml-replica/blob/main/mlreplica/utils/data.py), [algorithm](https://github.com/xianglous/ml-replica/blob/main/mlreplica/utils/algorithm.py), [loss](https://github.com/xianglous/ml-replica/blob/main/mlreplica/utils/loss.py), [metrics](https://github.com/xianglous/ml-replica/blob/main/mlreplica/utils/metrics.py), etc. that will be used across different types of models. 

The goal of this project is **NOT** to implement a super-efficient library for machine learning tasks. Rather, I created this repository to revisit the knowledge I acquired through college study and self-learning. I hope the repository can serve as a reference book for people in need of implementations or explanations. Therefore, the library will try to achieve simplicity and readability. In addition to the code, I add introduction to the models and algorithms on this website. Pseudocode and mathematical formulas are included to show the theoretical basis of the models.

The building of this repository is still **IN PROGRESS**. I am also learning new stuffs in the meantime. If you have any suggestions, or have identified any bugs or incorrect documentation, please feel free to contact me. 

*Enjoy!*