# second_order_methods.py

import tensorflow as tf

class SecondOrderMethods:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.hessian_approx = None

    def exp_concave_functions(self, model, inputs, targets, loss_fn):
        """
        Apply exponential weighting to exp-concave functions.
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
        exp_weights = tf.exp(-self.learning_rate * loss)
        weighted_gradients = [exp_weights * g for g in gradients]
        return weighted_gradients

    def exponentially_weighted_oco(self, model, inputs, targets, loss_fn):
        """
        Perform exponentially weighted online convex optimization.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :return: Weighted loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        if self.hessian_approx is None:
            self.hessian_approx = [tf.eye(g.shape[0]) for g in gradients]
        exp_weights = tf.exp(-self.learning_rate * loss)
        weighted_loss = exp_weights * loss
        return weighted_loss

    def online_newton_step(self, model, inputs, targets, loss_fn):
        """
        Perform the Online Newton Step Algorithm.
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
        if self.hessian_approx is None:
            self.hessian_approx = [tf.eye(g.shape[0]) for g in gradients]

        # Update Hessian approximation
        for i, grad in enumerate(gradients):
            grad = tf.reshape(grad, (-1, 1))
            self.hessian_approx[i] += tf.matmul(grad, grad, transpose_b=True)

        # Compute Newton step
        newton_step = [tf.linalg.solve(self.hessian_approx[i], g) for i, g in enumerate(gradients)]
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(newton_step, model.trainable_variables))
        return loss


