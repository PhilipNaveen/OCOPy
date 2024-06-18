# Machine Learning Algorithms API Documentation

## Overview

OCOPy is largely the same as OCOBox, a MATLAB toolbox made spefifically on the online convex optimization framework. OCOPy is a Python API that does the same, and is built on TensorFlow.

## Modules

### Gradient Descent

#### `gradient_descent.py`

- **batch_gradient_descent**: Performs gradient descent using the entire dataset to compute gradients.
- **mini_batch_gradient_descent**: Uses a subset of the dataset (mini-batch) to compute gradients for faster convergence.
- **stochastic_gradient_descent**: Updates model parameters after computing the gradient from a single random data point.

### Second Order Methods

#### `second_order_methods.py`

- **newton_method**: Optimization method using Newton's method for finding local minima.
- **online_newton_step**: Incremental version of Newton's method for online convex optimization.

### Regularization

#### `regularization.py`

- **rftl_algorithm**: Regularized Follow-The-Leader algorithm for online convex optimization.
- **online_mirror_descent**: Optimization method using mirror descent with regularization.

### Bandit Optimization

#### `bandit_optimization.py`

- **exp3_algorithm**: Exploration and exploitation algorithm for multi-armed bandits.
- **optimal_regret_algorithm**: Optimal regret algorithm for bandit linear optimization.

### Projection-Free Algorithms

#### `projection_free_algorithms.py`

- **conditional_gradient**: Optimization method using conditional gradient (Frank-Wolfe algorithm).
- **online_conditional_gradient**: Online version of conditional gradient method for linear optimization.

### Game Theory

#### `game_theory.py`

- **approximating_linear_programs**: Approximation of linear programs using game theory.
- **linear_programming_duality**: Implementation of linear programming duality theorem.

### Learning Theory

#### `learning_theory.py`

- **agnostic_learning**: Learning using online convex optimization in agnostic learning scenarios.
- **learning_and_compression**: Application of online convex optimization in learning and compression.

### Adaptive Regret

#### `adaptive_regret.py`

- **adaptive_regret_algorithm**: Algorithms for tracking adaptive regret in changing environments.
- **computationally_efficient_methods**: Efficient methods for adaptive regret minimization.

### Boosting

#### `boosting.py`

- **adaboost_algorithm**: Boosting algorithm using online convex optimization.
- **online_boosting**: Online version of boosting for sequential learning tasks.

### Blackwell Approachability

#### `blackwell_approachability.py`

- **vector_valued_games**: Vector-valued games and approachability concepts.
- **approachability_to_online_optimization**: Conversion of approachability to online convex optimization.

### Momentum Optimization

#### `momentum_optimization.py`

- **adam_optimizer**: Adaptive Moment Estimation (Adam) optimizer.
- **rmsprop_optimizer**: Root Mean Square Propagation (RMSProp) optimizer.

### Stepsize Scheduling

#### `stepsize_scheduling.py`

- **cosine_annealing_scheduler**: Cosine annealing learning rate scheduler.
- **exponential_decay_scheduler**: Exponential decay learning rate scheduler.

### Random Counter Examples

#### `random_counter_examples.py`

- **arbitrary_learning**: Arbitrary learning algorithm using a hypothesis class and teacher's responses.
- **majority_learning**: Majority learning algorithm using a hypothesis class and teacher's responses.
- **randomized_learning**: Randomized learning algorithm using a hypothesis class and teacher's responses.

## Utilities

#### `utils`

- **cuda_support.py**: Utilities for checking and enabling CUDA support.
- **helper_functions.py**: Miscellaneous helper functions used across the modules.

## Usage

Import the desired module and use the functions/classes as needed. Example:

```python
from gradient_descent import stochastic_gradient_descent
from utils.cuda_support import check_cuda_availability

# Example usage
params = stochastic_gradient_descent(initial_params, data, labels, learning_rate=0.001)
print("CUDA Available:", check_cuda_availability())
