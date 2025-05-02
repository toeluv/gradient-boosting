from methods.ada_delta import AdaDelta
from methods.adam import Adam
from methods.classic import Classic
from methods.momentum import Momentum
from methods.nag import NAG
from methods.rms_prop import RMSProp
from util.aggregating_results import aggregate_results
from util.plotting import create_boxplot

w1_coeffs, w2_coeffs = ([5, -4], [2, 5])

gradient_methods = [
    Classic(w1_coeffs, w2_coeffs),
    Momentum(w1_coeffs, w2_coeffs),
    NAG(w1_coeffs, w2_coeffs),
    RMSProp(w1_coeffs, w2_coeffs),
    AdaDelta(w1_coeffs, w2_coeffs),
    Adam(w1_coeffs, w2_coeffs)
]


def run():
    results = aggregate_results(gradient_methods, 500)
    create_boxplot(results).show()


if __name__ == '__main__':
    run()
