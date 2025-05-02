import random

from methods.gradient_method import GradientMethod


class Momentum(GradientMethod):
    def __init__(self, w1_coeffs, w2_coeffs, rate=0.01, epsilon=1e-6, max_iter=10000, lambda_=0.1):
        super().__init__(w1_coeffs, w2_coeffs, epsilon, max_iter)
        self.rate = rate
        self.gamma = 1 - lambda_
        self.eta = self.gamma * rate

    def run(self, initial_w=None):
        current_w = initial_w if initial_w else [random.uniform(-1, 1), random.uniform(-1, 1)]
        current_v = [0.0, 0.0]
        previous_w = current_w.copy()
        iteration = 0

        while iteration < self.max_iter:
            if self.is_done(previous_w, current_w, iteration):
                return iteration

            previous_w = current_w.copy()

            grad = self.calculate_gradient(current_w)

            current_v = [self.gamma * current_v[0] + self.eta * grad[0],
                         self.gamma * current_v[1] + self.eta * grad[1]]

            current_w = [current_w[0] - current_v[0],
                         current_w[1] - current_v[1]]

            iteration += 1

        return -1
