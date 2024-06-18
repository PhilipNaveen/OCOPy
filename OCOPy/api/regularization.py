# regularization.py

import tensorflow as tf

class RegularizationMethods:
    def __init__(self, learning_rate=0.01, regularization_fn=None):
        self.learning_rate = learning_rate
        self.regularization_fn = regularization_fn

    def rftl(self, model, inputs, targets, loss_fn):
        """
        Regularized Follow-The-Leader (RFTL) Algorithm.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
            if self.regularization_fn:
                loss += self.regularization_fn(model)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def online_mirror_descent(self, model, inputs, targets, loss_fn, mirror_map):
        """
        Online Mirror Descent (OMD) Algorithm.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param mirror_map: Mirror map function.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
            if self.regularization_fn:
                loss += self.regularization_fn(model)
        gradients = tape.gradient(loss, model.trainable_variables)
        transformed_gradients = [mirror_map(g) for g in gradients]
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(transformed_gradients, model.trainable_variables))
        return loss

    def adaptive_gradient_descent(self, model, inputs, targets, loss_fn, grad_squared_sum):
        """
        Adaptive Gradient Descent Algorithm.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param grad_squared_sum: Sum of squared gradients for adaptive updates.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
            if self.regularization_fn:
                loss += self.regularization_fn(model)
        gradients = tape.gradient(loss, model.trainable_variables)
        for i, grad in enumerate(gradients):
            grad_squared_sum[i] += tf.square(grad)
            adjusted_grad = grad / (tf.sqrt(grad_squared_sum[i]) + 1e-8)
            gradients[i] = adjusted_grad
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def perturbation_for_convex_losses(self, model, inputs, targets, loss_fn, noise_scale=0.1):
        """
        Randomized Regularization using perturbation for convex losses.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param noise_scale: Scale of the perturbation noise.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
            if self.regularization_fn:
                loss += self.regularization_fn(model)
            noise = [tf.random.normal(shape=tf.shape(var), stddev=noise_scale) for var in model.trainable_variables]
            loss += tf.reduce_sum([tf.reduce_sum(n * v) for n, v in zip(noise, model.trainable_variables)])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def perturbation_for_linear_cost_functions(self, model, inputs, targets, loss_fn, noise_scale=0.1):
        """
        Randomized Regularization using perturbation for linear cost functions.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param noise_scale: Scale of the perturbation noise.
        :return: Loss value.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)
            if self.regularization_fn:
                loss += self.regularization_fn(model)
            noise = [tf.random.normal(shape=tf.shape(var), stddev=noise_scale) for var in model.trainable_variables]
            loss += tf.reduce_sum([tf.reduce_sum(n * v) for n, v in zip(noise, model.trainable_variables)])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

