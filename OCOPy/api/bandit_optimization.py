# bandit_optimization.py

import tensorflow as tf
import numpy as np

class BanditOptimization:
    def __init__(self, learning_rate=0.01, num_actions=10, exploration_factor=0.1):
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.exploration_factor = exploration_factor
        self.action_values = tf.Variable(tf.zeros(num_actions), trainable=False)
        self.action_counts = tf.Variable(tf.zeros(num_actions), trainable=False)

    def exp3(self, rewards):
        """
        EXP3 Algorithm for multi-armed bandits.
        :param rewards: Reward vector for each action.
        :return: Selected action.
        """
        probabilities = tf.nn.softmax(self.action_values)
        selected_action = np.random.choice(self.num_actions, p=probabilities.numpy())
        reward = rewards[selected_action]

        estimated_reward = reward / probabilities[selected_action]
        new_action_value = self.action_values[selected_action] + self.learning_rate * estimated_reward
        self.action_values.scatter_nd_update([[selected_action]], [new_action_value])
        return selected_action

    def online_gradient_descent_without_gradient(self, model, inputs, targets, loss_fn, num_iterations=100, noise_scale=0.1):
        """
        Online Gradient Descent without a Gradient.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param num_iterations: Number of iterations for optimization.
        :param noise_scale: Scale of the perturbation noise.
        :return: Loss value.
        """
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)

        for _ in range(num_iterations):
            perturbations = [tf.random.normal(shape=tf.shape(var), stddev=noise_scale) for var in model.trainable_variables]
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = loss_fn(targets, predictions)
                perturbation_loss = tf.reduce_sum([tf.reduce_sum(p * v) for p, v in zip(perturbations, model.trainable_variables)])
                total_loss = loss + perturbation_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def optimize(self, model, inputs, targets, loss_fn, num_iterations=100):
        """
        Main optimization function.
        :param model: TensorFlow model.
        :param inputs: Input data.
        :param targets: Target data.
        :param loss_fn: Loss function.
        :param num_iterations: Number of iterations for optimization.
        :return: Loss value.
        """
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)

        for _ in range(num_iterations):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = loss_fn(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


