import abc
from abc import abstractmethod

import sympy as sp

w1, w2 = sp.symbols("w1, w2")


class GradientMethod(abc.ABC):
    def __init__(self, w1_coeffs, w2_coeffs, epsilon=1e-6, max_iter=10000):
        raw_target = w1_coeffs[0] * w1 * w1 + w1_coeffs[1] * w1 + w2_coeffs[0] * w2 * w2 + w2_coeffs[1] * w2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.target_func = sp.lambdify((w1, w2), raw_target)
        self.df_dw1 = sp.lambdify((w1, w2), raw_target.diff(w1))
        self.df_dw2 = sp.lambdify((w1, w2), raw_target.diff(w2))

    def calculate_gradient(self, w):
        return [self.df_dw1(w[0], w[1]), self.df_dw2(w[0], w[1])]

    def calculate_target(self, w):
        return self.target_func(w[0], w[1])

    def is_done(self, prev_w, curr_w, iteration):

        if iteration >= self.max_iter:
            print(f"Достигнуто максимальное число итераций: {self.max_iter}")
            return True

        if iteration == 0:
            return False

        max_diff = max(abs(curr_w[0] - prev_w[0]),
                       abs(curr_w[1] - prev_w[1]))

        if max_diff < self.epsilon:
            print(f"Решение найдено на итерации {iteration}")
            print(f"Оптимальные параметры: w1 = {curr_w[0]:.6f}, w2 = {curr_w[1]:.6f}")
            print(f"Значение функции: {self.calculate_target(curr_w):.6f}")
            return True

        return False

    @abstractmethod
    def run(self, initial_w=None):
        pass

    def name(self):
        return self.__class__.__name__
