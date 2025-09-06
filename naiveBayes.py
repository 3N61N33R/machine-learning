"""naiveBayes.py"""

import numpy as np


def naiveBayes(classes, learner, parameterised_function, train_data):
    f = {}
    parameters = {}
    g = {}
    for class_value in classes:
        parameters[class_value] = {}
        f[class_value] = {}

        # Takes the features associated with datapoints in a class
        train_x = train_data[train_data[:, -1] == class_value][:, :-1]

        for feature in range(train_x.shape[1]):
            parameters[class_value][feature] = learner(train_x[:, feature])
            f[class_value][feature] = parameterised_function(
                parameters[class_value][feature]
            )

        def create_g(class_value):
            def g(test_data):
                unscaled_feature_likelihoods = np.array(
                    [
                        [
                            f[class_value][feature](test_data[point, feature])
                            for feature in range(test_data.shape[1])
                        ]
                        for point in range(test_data.shape[0])
                    ]
                )
                unscaled_point_likelihood = np.prod(
                    unscaled_feature_likelihoods, axis=1
                ).reshape(-1, 1)
                return unscaled_point_likelihood

            return g

        g[class_value] = create_g(class_value)

    return g


# classes = [0,1]
# def learner(train):
#     mu = np.mean(train)
#     sig = np.std(train)
#     return [mu,sig]
# def parameterised_function(parameters):
#     mu = parameters[0]
#     sig = parameters[1]
#     return lambda x: np.exp(-0.5*(x - mu)**2/(sig**2))
# train_data = np.array([[2.0, 4.0, 0.0], [1.0, 5.0, 0.0], [4.0, 2.0, 1.0], [6.0, 0.0, 1.0]])
# g = naiveBayes(classes, learner, parameterised_function, train_data)
# test_data = np.array([[2.0, 5.0], [3.0,3.0]])

# for class_value in classes:
#     print(g[class_value](test_data))
