# game_theory.py

import numpy as np

class GameTheoryAlgorithms:
    def __init__(self):
        pass

    def approximating_linear_programs(self, constraints_matrix, objective_vector):
        """
        Approximating Linear Programs.
        :param constraints_matrix: Matrix representing the constraints.
        :param objective_vector: Vector representing the objective function.
        :return: Optimal solution vector.
        """
        # Solve the linear program approximation
        solution = np.linalg.lstsq(constraints_matrix, objective_vector, rcond=None)[0]
        return solution

    def linear_programming_duality(self, constraint_matrix, objective_vector):
        """
        Linear Programming Duality.
        :param constraint_matrix: Matrix representing the constraints.
        :param objective_vector: Vector representing the objective function.
        :return: Dual variables (Lagrange multipliers).
        """
        # Implement the duality theorem for linear programming
        dual_variables = np.linalg.inv(constraint_matrix.T) @ objective_vector
        return dual_variables

    def von_neumann_theorem(self, payoff_matrix):
        """
        Von Neumann Theorem for Zero-Sum Games.
        :param payoff_matrix: Payoff matrix of the zero-sum game.
        :return: Optimal mixed strategy for Player 1 and value of the game.
        """
        # Implement Von Neumann theorem for zero-sum games
        num_rows, num_cols = payoff_matrix.shape

        # Ensure it's a 2-player zero-sum game
        assert num_rows == num_cols, "Payoff matrix must be square."

        # Solve for Player 1's optimal mixed strategy
        value_of_game = np.min(np.max(payoff_matrix, axis=1))
        optimal_mixed_strategy = np.argmin(np.max(payoff_matrix, axis=1))

        return optimal_mixed_strategy, value_of_game

    def zero_sum_game(self, payoff_matrix):
        """
        Solving Zero-Sum Games.
        :param payoff_matrix: Payoff matrix of the zero-sum game.
        :return: Nash equilibrium strategies for both players.
        """
        # Implement solving zero-sum games
        num_rows, num_cols = payoff_matrix.shape

        # Ensure it's a 2-player zero-sum game
        assert num_rows == num_cols, "Payoff matrix must be square."

        # Solve for Nash equilibrium strategies using linear programming approach
        max_min_strategy = np.max(payoff_matrix, axis=0)
        min_max_strategy = np.min(payoff_matrix, axis=1)

        nash_eq_player1 = np.where(min_max_strategy == np.max(min_max_strategy))[0]
        nash_eq_player2 = np.where(max_min_strategy == np.min(max_min_strategy))[0]

        return nash_eq_player1, nash_eq_player2

