import duckdb
import os
import pytest
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import time

np.random.seed(42)

OBSERVATIONS = 1000
FEATURE_COUNT = 50
WEIGHTS = np.random.randint(-25, 25, size=FEATURE_COUNT)


for i in range(OBSERVATIONS):
    features = np.random.standard_normal(FEATURE_COUNT)
    label = np.dot(features, WEIGHTS)
    with open("rings.sql", "a") as f:
        f.write(f"INSERT INTO test VALUES ({list(features)}, {label});\n")



# Get a fresh connection to DuckDB with the quackml extension binary loaded
"""
@pytest.fixture
def duckdb_conn():
    extension_binary = os.getenv('QUACKML_EXTENSION_BINARY_PATH')
    if (extension_binary == ''):
        raise Exception('Please make sure the `QUACKML_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect('', config={'allow_unsigned_extensions': 'true'})
    conn.execute(f"load '{extension_binary}'")
    return conn

def test_quackml(duckdb_conn):
    duckdb_conn.execute("SELECT quackml('Sam') as value;");
    res = duckdb_conn.fetchall()
    assert(res[0][0] == "Quackml Sam 🐥");

def test_quackml_openssl_version_test(duckdb_conn):
    duckdb_conn.execute("SELECT quackml_openssl_version('Michael');");
    res = duckdb_conn.fetchall()
    assert(res[0][0][0:51] == "Quackml Michael, my linked OpenSSL version is OpenSSL");
"""

def main():
    conn = duckdb.connect()
    for line in open("rings.sql").readlines():
        conn.execute(line)

    conn.execute("COPY test TO 'output.csv' (HEADER, DELIMITER ',');")

    # Load into df, extract features and labels
    start = time.time()

    df = pd.read_csv('output.csv', header=None, names=["features", "label"])
    X = []
    for i in df["features"][1:]:
        row = []
        for j in i[1:-1].split(","):
            row.append(float(j))
        X.append(row)
    y = []
    for i in df["label"][1:]:
        y.append(float(i))

    end = time.time()    
    print(f"Import export time (ms): {(end - start) * 1000}")
    start = time.time()

    model = LinearRegression()
    model.fit(X, y)
    print(model.coef_)

    end = time.time() 
    print(f"Training time (ms): {(end - start) * 1000}")

if __name__ == "__main__":
    main()