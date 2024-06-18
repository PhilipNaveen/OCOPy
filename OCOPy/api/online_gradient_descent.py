# online_gradient_descent.py

import tensorflow as tf

class OnlineGradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def online_gradient_descent(self, model, inputs, targets, loss_fn):
        """
        Perform online gradient descent optimization.
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

    def lower_bounds(self, gradients, dimension):
        """
        Compute the lower bounds for online optimization.
        :param gradients: Gradient tensor.
        :param dimension: Dimensionality of the data.
        :return: Lower bound value.
        """
        return tf.reduce_sum(tf.square(gradients)) / (2 * dimension)

    def logarithmic_regret(self, model, inputs, targets, loss_fn, iterations):
        """
        Perform online gradient descent with logarithmic regret.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param iterations: Number of iterations.
        :return: Total regret.
        """
        total_regret = 0.0
        for t in range(1, iterations + 1):
            learning_rate = self.learning_rate / tf.sqrt(float(t))
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = loss_fn(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            regret = loss - tf.reduce_min(loss)
            total_regret += regret
        return total_regret

    def stochastic_gradient_descent(self, model, dataset, loss_fn, iterations):
        """
        Perform stochastic gradient descent optimization.
        :param model: TensorFlow model.
        :param dataset: TensorFlow dataset.
        :param loss_fn: Loss function.
        :param iterations: Number of iterations.
        :return: Trained model.
        """
        for inputs, targets in dataset.take(iterations):
            self.online_gradient_descent(model, inputs, targets, loss_fn)
        return model

    def sgd_for_svm(self, model, dataset, loss_fn, iterations):
        """
        Example: Stochastic Gradient Descent for SVM training.
        :param model: TensorFlow model.
        :param dataset: TensorFlow dataset.
        :param loss_fn: Loss function.
        :param iterations: Number of iterations.
        :return: Trained model.
        """
        return self.stochastic_gradient_descent(model, dataset, loss_fn, iterations)
