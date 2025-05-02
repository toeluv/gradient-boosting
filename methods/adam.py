import math
import random

from methods.gradient_method import GradientMethod


class Adam(GradientMethod):
    def __init__(self, w1_coeffs, w2_coeffs, rate=0.01, epsilon=1e-6, max_iter=10000,
                 lambda_=0.1, delta=0.01, alpha=0.999):
        super().__init__(w1_coeffs, w2_coeffs, epsilon, max_iter)
        self.rate = rate
        self.lambda_ = lambda_
        self.delta = delta
        self.alpha = alpha
        self.gamma = 1 - lambda_  # Коэффициент забывания

    def run(self, initial_w=None):
        current_w = initial_w if initial_w else [random.uniform(-1, 1), random.uniform(-1, 1)]
        V = [0.0, 0.0]
        G = [0.0, 0.0]
        previous_w = current_w.copy()
        iteration = 0

        while iteration < self.max_iter:
            if self.is_done(previous_w, current_w, iteration):
                return iteration

            previous_w = current_w.copy()

            grad = self.calculate_gradient(current_w)

            V = [
                self.gamma * V[0] + (1 - self.gamma) * grad[0],
                self.gamma * V[1] + (1 - self.gamma) * grad[1]
            ]

            G = [
                self.alpha * G[0] + (1 - self.alpha) * grad[0] * grad[0],
                self.alpha * G[1] + (1 - self.alpha) * grad[1] * grad[1]
            ]

            v_corrected = [
                V[0] / (1 - (self.gamma ** (iteration + 1))),
                V[1] / (1 - (self.gamma ** (iteration + 1)))
            ]

            g_corrected = [
                G[0] / (1 - (self.alpha ** (iteration + 1))),
                G[1] / (1 - (self.alpha ** (iteration + 1)))
            ]

            current_w = [
                current_w[0] - self.rate * v_corrected[0] / (math.sqrt(g_corrected[0]) + self.delta),
                current_w[1] - self.rate * v_corrected[1] / (math.sqrt(g_corrected[1]) + self.delta)
            ]

            iteration += 1

        return -1
