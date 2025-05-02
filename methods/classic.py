import random

from methods.gradient_method import GradientMethod


class Classic(GradientMethod):
    def __init__(self, w1_coeffs, w2_coeffs, rate=0.01, epsilon=1e-6, max_iter=10000):
        super().__init__(w1_coeffs, w2_coeffs, epsilon, max_iter)
        self.rate = rate

    def run(self, initial_w=None):
        current_w = initial_w if initial_w else [random.uniform(-1, 1), random.uniform(-1, 1)]
        iteration = 0
        previous_w = initial_w

        while iteration < self.max_iter:
            if self.is_done(previous_w, current_w, iteration):
                return iteration

            previous_w = current_w.copy()

            grad_w = self.calculate_gradient(current_w)
            current_w = [current_w[0] - grad_w[0] * self.rate, current_w[1] - grad_w[1] * self.rate]

            iteration += 1

        return -1
