from sklearn.linear_model import Ridge
import pandas as pd 
import duckdb
import time
import os

"""
What needs to be timed:
1. Export from DuckDB to csv 
2. Import from csv to sklearn 
3. Training of model in sklearn

NOT: Cleaning of data
"""

def test_linear_regression_benchmark(filename):
    conn = duckdb.connect()
    query = "CREATE TABLE tsv AS SELECT * FROM read_csv('" + filename + "', header=TRUE, delim='\t', columns={'features': 'DOUBLE[]', 'label': 'DOUBLE'})"
    conn.execute(query)

    start = time.time()
    # Export from DuckDB to csv (timed)
    conn.execute("COPY tsv TO 'test/quackml/export.csv' (HEADER, DELIMITER ',')")

    # Import from csv to sklearn (timed)
    data = pd.read_csv('test/quackml/export.csv')
    result = f"===== Test file: {filename} =====\n"
    result += f"Export/import time: {time.time() - start} seconds\n"
    cleaning_start = time.time()
    X = data['features'].apply(eval)
    features = pd.DataFrame(X.tolist(), dtype=float)
    y = data['label']
    cleaning_time = time.time() - cleaning_start

    result += f"Cleaning time: {cleaning_time} seconds\n"

    # Train model in sklearn (timed)
    train_start = time.time()
    model = Ridge(alpha=0.0, fit_intercept=True)
    model.fit(features, y)
    result += f"Training time: {time.time() - train_start} seconds\n"
    end = time.time()
    result += f"Total benchmark time: {(end - start) - cleaning_time} seconds\n\n"

    with open('test/quackml/benchmark_results.txt', 'a') as f:
        f.write(result)

# Example usage
def main():
    if os.path.exists('test/quackml/benchmark_results.txt'):
        os.remove('test/quackml/benchmark_results.txt')

    test_linear_regression_benchmark('test/quackml/test_100.tsv')
    test_linear_regression_benchmark('test/quackml/test_1000.tsv')
    test_linear_regression_benchmark('test/quackml/test_10000.tsv')
    test_linear_regression_benchmark('test/quackml/test_100000.tsv')
    test_linear_regression_benchmark('test/quackml/test_1000000.tsv')

if __name__ == "__main__":
    main()