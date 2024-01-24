import numpy as np 

weights = [
    [5, 3, -7, 8, 2, 3, 5, -1, -7, -9],
    [1, -4, 8, 9, 2, -5, -7, 9, 0, 4],
    [0, 9, 6, 2, -5, 1, -1, -8, 10, 2]
]
OBSERVATIONS = 1000
features = np.random.randn(OBSERVATIONS, len(weights[0]))
for f in features:
    feature_class = np.random.randint(0, 3)
    label = np.dot(f, weights[feature_class])
    with open("test/quackml/lin_reg_test.sql", "a") as sql_file:
        sql_file.write(f"INSERT INTO t VALUES ({list(f)}, {label}, {feature_class}); \n")