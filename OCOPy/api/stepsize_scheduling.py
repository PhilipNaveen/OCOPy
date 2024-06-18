# stepsize_scheduling.py

import numpy as np

class StepSizeSchedulers:
    def __init__(self):
        pass

    def constant_scheduler(self, initial_learning_rate):
        """
        Constant learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :return: Learning rate.
        """
        return initial_learning_rate

    def cosine_annealing_scheduler(self, initial_learning_rate, current_epoch, total_epochs):
        """
        Cosine annealing learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param total_epochs: Total number of epochs.
        :return: Learning rate.
        """
        return initial_learning_rate * 0.5 * (1 + np.cos(np.pi * current_epoch / total_epochs))

    def cosine_annealing_warm_restarts_scheduler(self, initial_learning_rate, current_epoch, T_0=10, T_mult=2):
        """
        Cosine annealing with warm restarts learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param T_0: Initial period.
        :param T_mult: Multiplicative factor for period increase.
        :return: Learning rate.
        """
        epoch_in_current_cycle = current_epoch % T_0
        return initial_learning_rate * 0.5 * (1 + np.cos(np.pi * epoch_in_current_cycle / T_0))

    def cyclic_scheduler(self, initial_learning_rate, current_epoch, cycle_length=10, decrease_factor=0.5):
        """
        Cyclic learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param cycle_length: Length of cycle.
        :param decrease_factor: Factor to decrease learning rate.
        :return: Learning rate.
        """
        return initial_learning_rate * (decrease_factor ** (current_epoch // cycle_length))

    def cyclical_log_annealing_scheduler(self, initial_learning_rate, current_epoch, total_epochs, T_mul=1.5):
        """
        Cyclical logarithmic annealing learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param total_epochs: Total number of epochs.
        :param T_mul: Multiplicative factor for period increase.
        :return: Learning rate.
        """
        T = total_epochs * (T_mul - 1) / np.log(T_mul)
        return initial_learning_rate * (np.log(T_mul) / (1 - np.log(current_epoch / T)))

    def exponential_decay_scheduler(self, initial_learning_rate, current_epoch, decay_rate=0.9):
        """
        Exponential decay learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param decay_rate: Decay rate.
        :return: Learning rate.
        """
        return initial_learning_rate * (decay_rate ** current_epoch)

    def inverse_time_decay_scheduler(self, initial_learning_rate, current_epoch, decay_rate=0.5, decay_step=1):
        """
        Inverse time decay learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param decay_rate: Decay rate.
        :param decay_step: Step for decay.
        :return: Learning rate.
        """
        return initial_learning_rate / (1 + decay_rate * (current_epoch // decay_step))

    def one_cycle_scheduler(self, initial_learning_rate, current_epoch, total_epochs, pct_start=0.3, div_factor=25, final_div_factor=1e4):
        """
        One cycle learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param total_epochs: Total number of epochs.
        :param pct_start: Percentage of cycle for annealing.
        :param div_factor: Factor for learning rate increase.
        :param final_div_factor: Final factor for learning rate decrease.
        :return: Learning rate.
        """
        cycle_epochs = int(total_epochs * pct_start)
        progress = min(current_epoch / cycle_epochs, 1)
        lr_ratio = final_div_factor / div_factor
        lr_diff = (initial_learning_rate / lr_ratio) - initial_learning_rate
        return initial_learning_rate + lr_diff * (1 + np.cos(np.pi * progress)) / 2

    def polynomial_decay_scheduler(self, initial_learning_rate, current_epoch, total_epochs, power=1.0):
        """
        Polynomial decay learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param total_epochs: Total number of epochs.
        :param power: Power factor.
        :return: Learning rate.
        """
        return initial_learning_rate * (1 - current_epoch / total_epochs) ** power

    def step_decay_scheduler(self, initial_learning_rate, current_epoch, drop_factor=0.5, epochs_drop=10):
        """
        Step decay learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param drop_factor: Factor to drop learning rate.
        :param epochs_drop: Number of epochs after which to drop learning rate.
        :return: Learning rate.
        """
        return initial_learning_rate * drop_factor ** (current_epoch // epochs_drop)

    def time_based_decay_scheduler(self, initial_learning_rate, current_epoch, decay_rate=0.1, decay_steps=1):
        """
        Time-based decay learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param decay_rate: Decay rate.
        :param decay_steps: Steps for decay.
        :return: Learning rate.
        """
        return initial_learning_rate / (1 + decay_rate * (current_epoch / decay_steps))

    def triangular_scheduler(self, initial_learning_rate, current_epoch, step_size=10, max_lr=1.0, method='triangular'):
        """
        Triangular learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param step_size: Step size for triangular cycle.
        :param max_lr: Maximum learning rate.
        :param method: Type of triangular cycle ('triangular' or 'triangular2').
        :return: Learning rate.
        """
        cycle = np.floor(1 + current_epoch / (2 * step_size))
        x = np.abs(current_epoch / step_size - 2 * cycle + 1)
        lr = initial_learning_rate + (max_lr - initial_learning_rate) * np.maximum(0, (1 - x))

        if method == 'triangular2':
            lr = lr * (1 / (2 ** (cycle - 1)))

        return lr

    def two_step_scheduler(self, initial_learning_rate, current_epoch, step_size=10, gamma=0.1):
        """
        Two-step learning rate scheduler.
        :param initial_learning_rate: Initial learning rate.
        :param current_epoch: Current epoch (starting from 1).
        :param step_size: Step size for decay.
        :param gamma: Gamma factor for decay.
        :return: Learning rate.
        """
        return initial_learning_rate * gamma ** (np.floor(current_epoch / step_size))

# Example usage:
if __name__ == "__main__":
    # Initialize the class
    schedulers = StepSizeSchedulers()

    # Example for Constant Scheduler
    initial_lr = 0.1
    print("Constant Scheduler Result:", schedulers.constant_scheduler(initial_lr))

    # Example for Cosine Annealing Scheduler
    current_epoch = 5
    total_epochs = 10
    print("Cosine Annealing Scheduler Result:", schedulers.cosine_annealing_scheduler(initial_lr, current_epoch, total_epochs))

    # Example for Cosine Annealing with Warm Restarts Scheduler
    T_0 = 5
    T_mult = 2
    print("Cosine Annealing with Warm Restarts Scheduler Result:", schedulers.cosine_annealing_warm_restarts_scheduler(initial_lr, current_epoch, T_0, T_mult))

    # Example for Cyclic Scheduler
    cycle_length = 5
    decrease_factor = 0.8
    print("Cyclic Scheduler Result:", schedulers.cyclic_scheduler(initial_lr, current_epoch, cycle_length, decrease_factor))

    # Example for Cyclical Log Annealing Scheduler
    T_mul = 1.5
    print("Cyclical Log Annealing Scheduler Result:", schedulers.cyclical_log_annealing_scheduler(initial_lr, current_epoch, total_epochs, T_mul))

    # Example for Exponential Decay Scheduler
    decay_rate = 0.9
    print("Exponential Decay Scheduler Result:", schedulers.exponential_decay_scheduler(initial_lr, current_epoch, decay_rate))

    # Example for Inverse Time Decay Scheduler
    decay_rate = 0.5
    decay_step = 2
    print("Inverse Time Decay Scheduler Result:", schedulers.inverse_time_decay_scheduler(initial_lr, current_epoch, decay_rate, decay_step))

    # Example for One Cycle Scheduler
    pct_start = 0.3
    div_factor = 20
    final_div_factor = 1000
    print("One Cycle Scheduler Result:", schedulers.one_cycle_scheduler(initial_lr, current_epoch, total_epochs, pct_start, div_factor, final_div_factor))

    # Example for Polynomial Decay Scheduler
    power = 0.9
    print("Polynomial Decay Scheduler Result:", schedulers.polynomial_decay_scheduler(initial_lr, current_epoch, total_epochs, power))

    # Example for Step Decay Scheduler
    drop_factor = 0.5
    epochs_drop = 5
    print("Step Decay Scheduler Result:", schedulers.step_decay_scheduler(initial_lr, current_epoch, drop_factor, epochs_drop))

    # Example for Time Based Decay Scheduler
    decay_rate = 0.1
    decay_steps = 2
    print("Time Based Decay Scheduler Result:", schedulers.time_based_decay_scheduler(initial_lr, current_epoch, decay_rate, decay_steps))

    # Example for Triangular Scheduler
    step_size = 5
    max_lr = 0.8
    method = 'triangular'
    print("Triangular Scheduler Result:", schedulers.triangular_scheduler(initial_lr, current_epoch, step_size, max_lr, method))

    # Example for Two Step Scheduler
    gamma = 0.9
    print("Two Step Scheduler Result:", schedulers.two_step_scheduler(initial_lr, current_epoch, step_size, gamma))
