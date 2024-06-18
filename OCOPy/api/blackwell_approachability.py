# blackwell_approachability.py

import numpy as np

class VectorValuedGames:
    def __init__(self):
        pass

    def approachability_reduction_to_oco(self, oracle, num_iterations=10):
        """
        Reduction from Approachability to Online Convex Optimization (OCO).
        :param oracle: Oracle for vector-valued games.
        :param num_iterations: Number of iterations.
        :return: Resulting vectors.
        """
        d = oracle.dimension
        eta = oracle.eta

        # Initialize vectors and weights
        V = np.zeros((num_iterations, d))
        weights = np.ones(num_iterations) / num_iterations

        for t in range(num_iterations):
            # Predict vector using oracle
            v_t = oracle.predict(weights)

            # Update weights
            weights *= np.exp(-eta * np.linalg.norm(v_t - V[t]))

            # Normalize weights
            weights /= np.sum(weights)

            # Store vector
            V[t] = v_t

        return V

    def reduction_to_approachability(self, OCO_algorithm, num_iterations=10):
        """
        Reduction from Online Convex Optimization (OCO) to Approachability.
        :param OCO_algorithm: Algorithm for Online Convex Optimization.
        :param num_iterations: Number of iterations.
        :return: Resulting oracle.
        """
        d = OCO_algorithm.dimension
        eta = OCO_algorithm.eta

        class ApproachabilityOracle:
            def __init__(self):
                self.dimension = d
                self.eta = eta

            def predict(self, weights):
                # Perform Online Convex Optimization
                x_t = OCO_algorithm.run(weights)

                # Return vector
                return x_t

        return ApproachabilityOracle()

    def vector_valued_game(self, strategies, payoff_matrix, num_iterations=10):
        """
        Vector-valued game with approachability.
        :param strategies: List of strategies for players.
        :param payoff_matrix: Payoff matrix for the game.
        :param num_iterations: Number of iterations.
        :return: Resulting strategies.
        """
        T, d = num_iterations, len(strategies)
        eta = 1.0 / T

        # Initialize strategies and weights
        S = np.zeros((T, d))
        weights = np.ones(T) / T

        for t in range(T):
            # Calculate average payoff
            avg_payoff = np.zeros(d)
            for i in range(d):
                avg_payoff[i] = np.dot(payoff_matrix[i], weights)

            # Update strategies
            S[t] = strategies[t] - eta * avg_payoff

            # Update weights
            weights *= np.exp(-eta * np.linalg.norm(S[t]))

            # Normalize weights
            weights /= np.sum(weights)

        return S

