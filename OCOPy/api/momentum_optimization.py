# momentum_optimization.py

import numpy as np

class MomentumOptimization:
    def __init__(self):
        pass

    def gradient_descent_with_momentum(self, initial_params, gradient_fn, learning_rate=0.01, momentum=0.9, num_iterations=100):
        """
        Gradient Descent with Momentum algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param momentum: Momentum parameter.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        velocity = np.zeros_like(initial_params)

        for t in range(num_iterations):
            gradient = gradient_fn(params)
            velocity = momentum * velocity + learning_rate * gradient
            params -= velocity

        return params

    def stochastic_gradient_descent_with_momentum(self, initial_params, gradient_fn, dataset, batch_size=32, learning_rate=0.01, momentum=0.9, num_epochs=10):
        """
        Stochastic Gradient Descent with Momentum algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param dataset: Dataset for training.
        :param batch_size: Batch size.
        :param learning_rate: Learning rate.
        :param momentum: Momentum parameter.
        :param num_epochs: Number of epochs.
        :return: Optimized parameters.
        """
        params = initial_params
        velocity = np.zeros_like(initial_params)
        num_batches = len(dataset) // batch_size

        for epoch in range(num_epochs):
            np.random.shuffle(dataset)
            for batch in range(num_batches):
                batch_data = dataset[batch * batch_size:(batch + 1) * batch_size]
                gradient = gradient_fn(params, batch_data)
                velocity = momentum * velocity + learning_rate * gradient
                params -= velocity

        return params

    def mini_batch_gradient_descent_with_momentum(self, initial_params, gradient_fn, dataset, batch_size=32, learning_rate=0.01, momentum=0.9, num_epochs=10):
        """
        Mini-Batch Gradient Descent with Momentum algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param dataset: Dataset for training.
        :param batch_size: Batch size.
        :param learning_rate: Learning rate.
        :param momentum: Momentum parameter.
        :param num_epochs: Number of epochs.
        :return: Optimized parameters.
        """
        params = initial_params
        velocity = np.zeros_like(initial_params)
        num_batches = len(dataset) // batch_size

        for epoch in range(num_epochs):
            np.random.shuffle(dataset)
            for batch in range(num_batches):
                batch_data = dataset[batch * batch_size:(batch + 1) * batch_size]
                gradient = gradient_fn(params, batch_data)
                velocity = momentum * velocity + learning_rate * gradient
                params -= velocity

        return params

    def nestrov_momentum(self, initial_params, gradient_fn, learning_rate=0.01, momentum=0.9, num_iterations=100):
        """
        Nesterov Momentum algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param momentum: Momentum parameter.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        velocity = np.zeros_like(initial_params)

        for t in range(num_iterations):
            gradient = gradient_fn(params + momentum * velocity)
            velocity = momentum * velocity + learning_rate * gradient
            params -= velocity

        return params

    def adagrad(self, initial_params, gradient_fn, learning_rate=0.01, epsilon=1e-8, num_iterations=100):
        """
        Adagrad algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        accumulated_gradients = np.zeros_like(initial_params) + epsilon

        for t in range(num_iterations):
            gradient = gradient_fn(params)
            accumulated_gradients += gradient ** 2
            params -= learning_rate * gradient / np.sqrt(accumulated_gradients)

        return params

    def rmsprop(self, initial_params, gradient_fn, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, num_iterations=100):
        """
        RMSprop algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param decay_rate: Decay rate for moving average.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        accumulated_gradients = np.zeros_like(initial_params) + epsilon

        for t in range(num_iterations):
            gradient = gradient_fn(params)
            accumulated_gradients = decay_rate * accumulated_gradients + (1 - decay_rate) * gradient ** 2
            params -= learning_rate * gradient / np.sqrt(accumulated_gradients)

        return params

    def adam(self, initial_params, gradient_fn, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=100):
        """
        Adam algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param beta1: Exponential decay rate for the 1st moment estimates.
        :param beta2: Exponential decay rate for the 2nd moment estimates.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        m = np.zeros_like(initial_params)
        v = np.zeros_like(initial_params)
        t = 0

        for t in range(num_iterations):
            t += 1
            gradient = gradient_fn(params)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        return params

    def amsgrad(self, initial_params, gradient_fn, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=100):
        """
        AMSGrad algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param beta1: Exponential decay rate for the 1st moment estimates.
        :param beta2: Exponential decay rate for the 2nd moment estimates.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        m = np.zeros_like(initial_params)
        v = np.zeros_like(initial_params)
        v_hat = np.zeros_like(initial_params)
        t = 0

        for t in range(num_iterations):
            t += 1
            gradient = gradient_fn(params)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            v_hat = np.maximum(v_hat, v)
            params -= learning_rate * m / (np.sqrt(v_hat) + epsilon)

        return params

    def adadelta(self, initial_params, gradient_fn, decay_rate=0.9, epsilon=1e-8, num_iterations=100):
        """
        Adadelta algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param decay_rate: Decay rate for moving average.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        accumulated_gradients = np.zeros_like(initial_params)
        accumulated_deltas = np.zeros_like(initial_params)
        update = np.zeros_like(initial_params)

        for t in range(num_iterations):
            gradient = gradient_fn(params)
            accumulated_gradients = decay_rate * accumulated_gradients + (1 - decay_rate) * gradient ** 2
            delta_params = - np.sqrt(accumulated_deltas + epsilon) * gradient / np.sqrt(accumulated_gradients + epsilon)
            params += delta_params
            accumulated_deltas = decay_rate * accumulated_deltas + (1 - decay_rate) * delta_params ** 2

        return params

    def eve(self, initial_params, gradient_fn, learning_rate=0.01, beta1=0.9, beta2=0.999, beta3=0.999, beta4=0.999, epsilon=1e-8, num_iterations=100):
        """
        Eve algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param learning_rate: Learning rate.
        :param beta1: Exponential decay rate for the 1st moment estimates.
        :param beta2: Exponential decay rate for the 2nd moment estimates.
        :param beta3: Exponential decay rate for the 3rd moment estimates.
        :param beta4: Exponential decay rate for the 4th moment estimates.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        m = np.zeros_like(initial_params)
        v = np.zeros_like(initial_params)
        t = 0

        for t in range(num_iterations):
            t += 1
            gradient = gradient_fn(params)
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            alpha_t = learning_rate / (1 - beta3 ** t)
            beta_t = 1 / (1 - beta4 ** t)
            params -= alpha_t * beta_t * m_hat / (np.sqrt(v_hat) + epsilon)

        return params

    def fasfa(self, initial_params, gradient_fn, mu=0.9, nu=0.999, alpha=0.01, epsilon=1e-8, num_iterations=100):
        """
        FASFA (Fast Adaptive Stochastic Function Acceleration) algorithm.
        :param initial_params: Initial parameters (weights).
        :param gradient_fn: Function to compute gradients.
        :param mu: First order momentum decay estimate.
        :param nu: Second order momentum decay estimate.
        :param alpha: Learning rate.
        :param epsilon: Small value to prevent division by zero.
        :param num_iterations: Number of iterations.
        :return: Optimized parameters.
        """
        params = initial_params
        m = np.zeros_like(initial_params)
        n = np.zeros_like(initial_params)
        t = 0

        for t in range(num_iterations):
            t += 1
            gradient = gradient_fn(params)
            m = mu * m + (1 - mu) * gradient
            n = nu * n + (1 - nu) * gradient ** 2
            m_hat = mu * m + gradient * (1 - mu)
            n_hat = nu * n + gradient ** 2 * (1 - nu)
            params -= alpha * m_hat * np.sqrt(1 - mu ** t) / (np.sqrt(n_hat) + epsilon)

        return params

