import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

class InteractiveLearningNN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def arbitrary_learning(self, teacher):
        """
        Arbitrary Learning Algorithm using Neural Network.
        :param teacher: Function to query the teacher for counter-examples.
        :return: Learned hypothesis h.
        """
        while True:
            # Train neural network on current hypotheses
            self.model.fit(self.H, self.y, epochs=10, batch_size=32, verbose=0)

            # Pick an arbitrary hypothesis (random from current model state)
            i = np.random.randint(len(self.H))
            hi = self.H[i]

            # Query the teacher
            xi = teacher(hi)

            # If no counter-example returned by teacher, output hypothesis
            if xi is None:
                return hi
            else:
                # Update dataset with new counter-example
                self.H = np.vstack([self.H, xi]); self.y = np.append(self.y, teacher(xi))

    def majority_learning(self, teacher):
        """
        Majority Learning Algorithm using Neural Network.
        :param teacher: Function to query the teacher for counter-examples.
        :return: Learned hypothesis h.
        """
        while True:
            # Train neural network on current hypotheses
            self.model.fit(self.H, self.y, epochs=10, batch_size=32, verbose=0)

            # Predict majority hypothesis from current model state
            h = self.majority_hypothesis()

            # Query the teacher
            x = teacher(h)

            # If no counter-example returned by teacher, output hypothesis
            if x is None:
                return h
            else:
                # Updaate dataset with new counter-example
                self.H = np.vstack([self.H, x])
                self.y = np.append(self.y, teacher(x))

    def randomized_learning(self, teacher):
        """
        Randomized Learning Algorithm using Neural Network.
        :param teacher: Function to query the teacher for counter-examples.
        :return: Learned hypothesis h.
        """
        r = 1
        while True:
            # Train neural network on current hypotheses
            self.model.fit(self.H, self.y, epochs=10, batch_size=32, verbose=0)

            # Draw a random hypothesis from current model state
            hr = self.random_hypothesis()

            # Query the teacher
            xr = teacher(hr)

            # If no counter-example returned by teacher, output hypothesis
            if xr is None:
                return hr
            else:
                # Update dataset with new counter-example
                self.H = np.vstack([self.H, xr])
                self.y = np.append(self.y, teacher(xr))
                # Implementing Qr+1 is not shown here as it depends on specific problem

    def majority_hypothesis(self):
        """
        Computes majority hypothesis based on current model predictions.
        :return: Majority hypothesis.
        """
        y_pred = self.model.predict(self.H)
        h = np.mean(self.H[y_pred >= 0.5], axis=0)
        return h

    def random_hypothesis(self):
        """
        Draws a random hypothesis from the current model state.
        :return: Random hypothesis.
        """
        i = np.random.randint(len(self.H))
        return self.H[i]

    def initialize(self, H, y):
        """
        Initializes the hypothesis class and labels for interactive learning.
        :param H: Initial hypothesis class (n x m matrix).
        :param y: Initial labels corresponding to H.
        """
        self.H = H
        self.y = y


