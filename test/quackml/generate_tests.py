import numpy as np 
import os

SEED = 42
DIMENSIONS = 25
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

def generate_group_test(fname: str, groups: int, n: int) -> None:
    weights = np.random.randint(-20, 20, size=(groups, DIMENSIONS))
    for weight in weights:
        all_weights.append(weight)
    features = np.random.standard_normal(size=(n, DIMENSIONS-1))
    
    if os.path.exists(fname):
        os.remove(fname)

    with open(fname, 'w') as f:
        f.write("features\tlabel\tclass\n")
        for feat in features:
            group = np.random.randint(0, groups)
            label = np.dot(feat, weights[group][:-1]) + weights[group][-1]
            f.write(f"{list(feat)}\t{label}\t{group}\n")

def generate_join_test(fname_a: str, fname_b: str, n: int) -> None:
    weights = np.random.randint(-20, 20, size=2)
    all_weights.append(weights)
    features = np.random.standard_normal(size=(n, 2))
    labels = np.dot(features, weights)
    ids = np.arange(n)

    all_data = {}
    for id_, feat, label in zip(ids, features, labels):
        all_data[id_] = (feat, label)

    if os.path.exists(fname_a):
        os.remove(fname_a)
    if os.path.exists(fname_b):
        os.remove(fname_b)

    # Shuffle data
    np.random.shuffle(ids)
    
    with open(fname_a, 'w') as f:
        f.write("label\tid\n")
        for id_ in ids:
            f.write(f"[{all_data[id_][1]}]\t{id_}\n")

    np.random.shuffle(ids)

    with open(fname_b, 'w') as f:
        f.write("features\tid\n")
        for id_ in ids:
            f.write(f"{list(all_data[id_][0])}\t{id_}\n")

def generate_join_group_test(fname_a: str, fname_b: str, groups: int, n: int) -> None:
    weights = np.random.randint(-20, 20, size=(groups, DIMENSIONS))
    for weight in weights:
        all_weights.append(weight)
    features = np.random.standard_normal(size=(n, DIMENSIONS))
    ids = np.arange(n)

    all_data = {}
    for id_, feat in zip(ids, features):
        group_ = np.random.randint(0, groups)
        label = np.dot(feat, weights[group_])
        all_data[id_] = (feat, group_, label)

    if os.path.exists(fname_a):
        os.remove(fname_a)
    if os.path.exists(fname_b):
        os.remove(fname_b)

    np.random.shuffle(ids)
    # Write label and id in fname_a
    with open(fname_a, 'w') as f:
        f.write("label\tid\n")
        for id_ in ids:
            f.write(f"[{all_data[id_][2]}]\t{id_}\n")

    np.random.shuffle(ids)
    # Write features, id, and group in fname_b
    with open(fname_b, 'w') as f:
        f.write("features\tid\tclass\n")
        for id_ in ids:
            f.write(f"{list(all_data[id_][0])}\t{id_}\t{all_data[id_][1]}\n")

def factorised_test(fname: str, n: int) -> None:
    # Generate n datasets containing 10 observations of 5 features
    for i in range(n):
        features = np.random.standard_normal(size=(10, 5))
        
        dataset_name = f"{fname}_{i}.tsv"
        if os.path.exists(dataset_name):
            os.remove(dataset_name)    

        with open(dataset_name, 'w') as f:
            f.write("features\tid\n")
            for feat in features:
                id_ = np.random.randint(0, 3)
                f.write(f"{list(feat)}\t{id_}\n")


def main():
    # for i in range(2, 21):
    #     size = 2**i
    #     fname = f"test/quackml/datasets/test_{i}.tsv"
    #     generate_test_set(fname, size)

    #generate_group_test('test/quackml/test_groups.tsv', 5, 1000)
    #generate_join_test('test/quackml/test_join_a.tsv', 'test/quackml/test_join_b.tsv', 5)
    #generate_join_group_test('test/quackml/test_join_group_a.tsv', 'test/quackml/test_join_group_b.tsv', 5, 1000)

    factorised_test('test/quackml/datasets/factorised', 10)

    if os.path.exists("test/quackml/weights.txt"):
        os.remove("test/quackml/weights.txt")

    with open("test/quackml/weights.txt", 'w') as f:
        for weights in all_weights:
            f.write(f"{list(weights)}\n")

if __name__ == "__main__":
    main()
