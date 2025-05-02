from methods.gradient_method import GradientMethod


def aggregate_results(methods: list[GradientMethod], counts):
    def multiply_tests(method: GradientMethod, count):
        result = []
        for _ in range(count):
            iterations = method.run()
            if iterations > 0:
                result.append(iterations)
        return result

    keys = [method.name() for method in methods]
    values = [multiply_tests(method, counts) for method in methods]
    return dict(zip(keys, values))
