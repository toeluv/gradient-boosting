import math
import random

from methods.gradient_method import GradientMethod


class AdaDelta(GradientMethod):
    def __init__(self,w1_coeffs, w2_coeffs, rate=0.01, epsilon=1e-6, max_iter=10000, delta=0.1, delta_b=0.1, alpha=0.999):
        super().__init__(w1_coeffs, w2_coeffs,epsilon, max_iter)
        self.rate = rate
        self.delta = delta
        self.delta_b = delta_b
        self.alpha = alpha

    def run(self, initial_w=None):
        current_w = initial_w if initial_w else [random.uniform(-1, 1), random.uniform(-1, 1)]
        current_v = [0.0, 0.0]
        previous_w = current_w.copy()
        iteration = 0
        delta_b = self.delta_b

        while iteration < self.max_iter:
            if self.is_done(previous_w, current_w, iteration):
                return iteration

            previous_w = current_w.copy()

            grad = self.calculate_gradient(current_w)

            current_v = [self.alpha * current_v[0] + (1 - self.alpha) * grad[0] * grad[0],
                         self.alpha * current_v[1] + (1 - self.alpha) * grad[1] * grad[1]]

            delta = [
                grad[0] * math.sqrt((delta_b + self.epsilon) / current_v[0] + self.epsilon),
                grad[1] * math.sqrt((delta_b + self.epsilon) / current_v[1] + self.epsilon)
            ]

            delta_b = self.alpha * delta_b + (1 - self.alpha) * self.delta * self.delta

            current_w = [current_w[0] - delta[0],
                         current_w[1] - delta[1]]

            iteration += 1

        return -1
