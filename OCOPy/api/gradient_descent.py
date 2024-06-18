# gradient_descent.py

import tensorflow as tf

class GradientDescent:
    def __init__(self, learning_rate=0.01, optimal_loss_value=0):
        self.learning_rate = learning_rate
        self.optimal_loss_value = optimal_loss_value

    def polyak_stepsize(self, gradient, loss_value):
        """
        Compute the Polyak stepsize.
        :param gradient: Gradient tensor.
        :param loss_value: Current loss value.
        :return: Polyak stepsize.
        """
        return (loss_value - self.optimal_loss_value) / (tf.reduce_sum(tf.square(gradient)) + 1e-8)

    def measure_distance_to_optimality(self, current_value, optimal_value):
        """
        Measure the distance to optimality.
        :param current_value: Current parameter value.
        :param optimal_value: Optimal parameter value.
        :return: Distance to optimality.
        """
        return tf.reduce_sum(tf.square(current_value - optimal_value))

    def basic_gradient_descent(self, model, inputs, targets, loss_fn):
        """
        Perform basic gradient descent optimization.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def constrained_gradient_descent(self, model, inputs, targets, loss_fn, constraints_fn):
        """
        Perform constrained gradient descent optimization.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param constraints_fn: Function to enforce constraints on model parameters.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        constraints_fn(model.trainable_variables)
        return loss

    def optimize_with_polyak(self, model, inputs, targets, loss_fn):
        """
        Optimize using Polyak stepsize.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        stepsize = self.polyak_stepsize(gradients, loss)
        optimizer = tf.optimizers.SGD(learning_rate=stepsize)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def optimize_smooth_non_strongly_convex(self, model, inputs, targets, loss_fn, smoothness_constant):
        """
        Optimize smooth, non-strongly convex functions.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param smoothness_constant: Smoothness constant (L).
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        stepsize = 1 / smoothness_constant
        optimizer = tf.optimizers.SGD(learning_rate=stepsize)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def optimize_strongly_convex_non_smooth(self, model, inputs, targets, loss_fn, strong_convexity_constant):
        """
        Optimize strongly convex, non-smooth functions.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param strong_convexity_constant: Strong convexity constant (mu).
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        stepsize = 1 / strong_convexity_constant
        optimizer = tf.optimizers.SGD(learning_rate=stepsize)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def optimize_general_convex(self, model, inputs, targets, loss_fn):
        """
        Optimize general convex functions.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def svd_training(self, model, inputs, targets, loss_fn, iterations=1000):
        """
        Train a Support Vector Machine (SVM) using gradient descent.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param iterations: Number of iterations.
        :return: Trained model.
        """
        for _ in range(iterations):
            self.basic_gradient_descent(model, inputs, targets, loss_fn)
        return model
