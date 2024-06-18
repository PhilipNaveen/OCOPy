# adaptive_regret.py

import numpy as np

class AdaptiveRegretAlgorithms:
    def __init__(self):
        pass

    def dynamic_regret(self, online_optimizer, loss_function, num_iterations=100):
        """
        Dynamic Regret in Online Convex Optimization.
        :param online_optimizer: Online convex optimizer object.
        :param loss_function: Loss function for optimization.
        :param num_iterations: Number of iterations for optimization.
        :return: Regret over time.
        """
        regret = []
        current_params = np.zeros(online_optimizer.dimension)
        
        for t in range(1, num_iterations + 1):
            # Receive loss function at time t
            loss_t = loss_function(current_params)
            
            # Optimize and update parameters
            current_params = online_optimizer.optimize(loss_t)
            
            # Calculate regret at time t
            optimal_loss = online_optimizer.minimize_cumulative_loss(loss_function, t)
            current_loss = loss_function(current_params)
            regret_t = current_loss - optimal_loss
            
            regret.append(regret_t)
        
        return regret

    def adaptive_regret(self, online_optimizer, loss_function, num_iterations=100):
        """
        Adaptive Regret in Online Convex Optimization.
        :param online_optimizer: Online convex optimizer object.
        :param loss_function: Loss function for optimization.
        :param num_iterations: Number of iterations for optimization.
        :return: Adaptive regret over time.
        """
        adaptive_regret = []
        current_params = np.zeros(online_optimizer.dimension)
        cum_loss = 0.0
        
        for t in range(1, num_iterations + 1):
            # Receive loss function at time t
            loss_t = loss_function(current_params)
            
            # Optimize and update parameters
            current_params = online_optimizer.optimize(loss_t)
            
            # Update cumulative loss
            cum_loss += loss_t
            
            # Calculate adaptive regret at time t
            optimal_loss = online_optimizer.minimize_cumulative_loss(loss_function, t)
            adaptive_regret_t = cum_loss - optimal_loss
            
            adaptive_regret.append(adaptive_regret_t)
        
        return adaptive_regret

