import numpy as np 
import os

SEED = 42
DIMENSIONS = 50

all_weights = []

def generate_test_set(fname: str, n: int) -> None:
    weights = np.random.randint(-20, 20, size=DIMENSIONS)
    all_weights.append(weights)
    features = np.random.standard_normal(size=(n, DIMENSIONS-1))
    labels = np.dot(features, weights[:-1]) + weights[-1]
    
    if os.path.exists(fname):
        os.remove(fname)    

    with open(fname, 'w') as f:
        f.write("features\tlabel\n")
        for feat, label in zip(features, labels):
            f.write(f"{list(feat)}\t{label}\n")

if __name__ == "__main__":
    generate_test_set('test/quackml/test_100.tsv', 100)
    generate_test_set('test/quackml/test_1000.tsv', 1000)
    generate_test_set('test/quackml/test_10000.tsv', 10000)
    generate_test_set('test/quackml/test_100000.tsv', 100000)
    generate_test_set('test/quackml/test_1000000.tsv', 1000000)

    if os.path.exists("test/quackml/weights.txt"):
        os.remove("test/quackml/weights.txt")

    with open("test/quackml/weights.txt", 'w') as f:
        for weights in all_weights:
            f.write(f"{list(weights)}\n")