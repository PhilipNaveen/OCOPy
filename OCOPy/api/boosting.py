# boosting.py

import numpy as np

class BoostingAlgorithms:
    def __init__(self):
        pass

    def adaboost(self, weak_learners, data_points, labels, num_iterations=10):
        """
        AdaBoost Algorithm.
        :param weak_learners: List of weak learners.
        :param data_points: Data points for training.
        :param labels: Labels for training.
        :param num_iterations: Number of boosting iterations.
        :return: Boosted model.
        """
        N = len(data_points)
        weights = np.ones(N) / N
        models = []
        alpha = []

        for t in range(num_iterations):
            # Train weak learner with weighted samples
            model_t = weak_learners[t].train(data_points, labels, weights)

            # Predictions and error calculation
            predictions = model_t.predict(data_points)
            error_t = np.sum(weights * (predictions != labels)) / np.sum(weights)

            # Compute alpha
            alpha_t = 0.5 * np.log((1 - error_t) / error_t)

            # Update weights
            weights = weights * np.exp(-alpha_t * labels * predictions)
            weights = weights / np.sum(weights)

            # Store model and alpha
            models.append(model_t)
            alpha.append(alpha_t)

        # Define boosted model class
        class BoostedModel:
            def __init__(self, models, alpha):
                self.models = models
                self.alpha = alpha

            def predict(self, x):
                predictions = np.zeros(len(x))
                for t in range(len(self.models)):
                    predictions += self.alpha[t] * self.models[t].predict(x)
                return np.sign(predictions)

        # Return boosted model
        return BoostedModel(models, alpha)

    def online_boosting(self, weak_learner, data_points, labels, num_iterations=10):
        """
        Online Boosting Algorithm.
        :param weak_learner: Weak learner for online boosting.
        :param data_points: Data points for training.
        :param labels: Labels for training.
        :param num_iterations: Number of boosting iterations.
        :return: Online boosted model.
        """
        N = len(data_points)
        weights = np.ones(N) / N
        models = []

        for t in range(num_iterations):
            # Train weak learner with weighted samples
            model_t = weak_learner.train(data_points, labels, weights)

            # Predictions and error calculation
            predictions = model_t.predict(data_points)
            error_t = np.sum(weights * (predictions != labels)) / np.sum(weights)

            # Update weights
            weights = weights * np.exp(-error_t * labels * predictions)
            weights = weights / np.sum(weights)

            # Store model
            models.append(model_t)

        # Define online boosted model class
        class OnlineBoostedModel:
            def __init__(self, models):
                self.models = models

            def predict(self, x):
                predictions = np.zeros(len(x))
                for t in range(len(self.models)):
                    predictions += self.models[t].predict(x)
                return np.sign(predictions)

        # Return online boosted model
        return OnlineBoostedModel(models)

    def boosting_by_oco(self, weak_learner, data_points, labels, num_iterations=10, learning_rate=0.1):
        """
        Boosting by Online Convex Optimization (OCO).
        :param weak_learner: Weak learner for boosting.
        :param data_points: Data points for training.
        :param labels: Labels for training.
        :param num_iterations: Number of boosting iterations.
        :param learning_rate: Learning rate for OCO.
        :return: Boosted model.
        """
        N = len(data_points)
        models = []
        weights = np.ones(N) / N

        for t in range(num_iterations):
            # Train weak learner with weighted samples
            model_t = weak_learner.train(data_points, labels, weights)

            # Predictions and error calculation
            predictions = model_t.predict(data_points)
            error_t = np.sum(weights * (predictions != labels)) / np.sum(weights)

            # Update weights using OCO update rule
            weights *= np.exp(-learning_rate * labels * predictions)

            # Normalize weights
            weights /= np.sum(weights)

            # Store model
            models.append(model_t)

        # Define OCO boosted model class
        class OCOBoostedModel:
            def __init__(self, models):
                self.models = models

            def predict(self, x):
                predictions = np.zeros(len(x))
                for t in range(len(self.models)):
                    predictions += self.models[t].predict(x)
                return np.sign(predictions)

        # Return OCO boosted model
        return OCOBoostedModel(models)

