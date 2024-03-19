from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd 
import numpy as np
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
    # Split the data into training/testing sets (80/20)
    training_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
    ridge_model = Ridge(alpha=0)
    ridge_model.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = ridge_model.predict(X_test)
    # Report RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    training_time = time.time() - training_start
    result += f"Training time: {training_time} seconds\n"
    result += f"RMSE: {rmse}\n"

    with open('test/quackml/benchmark_results.txt', 'a') as f:
        f.write(result)

def test_cartesian_product(filename, n):
    """ 
    Load n datasets (untimed)
    Clean to seperate columns (untimed)
    Compute join - Join time
    Export to csv - Import/export time
    Load into pandas - Import/export time
    Train model - Training time
    """
    conn = duckdb.connect()

    total_import_time = 0
    total_join_time = 0
    total_training_time = 0

    for i in range(n):
        database_name = f"test/quackml/datasets/factorised_{i}.tsv"
        table_name = f"tsv_{i}"
        query = f"CREATE TABLE {table_name} AS SELECT * FROM read_csv('{database_name}', "   
        query += "header=TRUE, delim='\t', columns={'features': 'DOUBLE[]', 'id': 'INTEGER'})"
        import_start = time.time()
        conn.execute(query)
        total_import_time += (time.time() - import_start)

        # Clean to seperate columns (untimed)
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN f1 DOUBLE;")
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN f2 DOUBLE;")
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN f3 DOUBLE;")
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN f4 DOUBLE;")
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN f5 DOUBLE;")
        conn.execute(f"UPDATE {table_name} SET f1 = features[1], f2 = features[2], f3 = features[3], f4 = features[4], f5 = features[5];")
        conn.execute(f"ALTER TABLE {table_name} DROP COLUMN features;")

    # Compute join
    join_query = f"CREATE TABLE joined AS SELECT * FROM tsv_0"
    for i in range(1, n):
        join_query += f" JOIN tsv_{i} ON tsv_0.id = tsv_{i}.id"

    join_query += ";"

    join_start = time.time()
    conn.execute(join_query)
    total_join_time += time.time() - join_start
    # conn.execute("ALTER TABLE joined DROP COLUMN id;")
    # for i in range(1, n):
    #     conn.execute(f"ALTER TABLE joined DROP COLUMN id_{i};")

    # Export to csv
    export_start = time.time()
    conn.execute("COPY joined TO 'test/quackml/export.csv' (HEADER, DELIMITER ',')")

    # Import from csv to sklearn
    data = pd.read_csv('test/quackml/export.csv')
    total_import_time += (time.time() - export_start)

    # Train model in sklearn, use first column as label
    train_start = time.time()
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    ridge_model = Ridge(alpha=0)
    ridge_model.fit(X, y)
    training_time = time.time() - train_start

    # Write results 
    with open('test/quackml/factorised_test_results.txt', 'a') as f:
        f.write(f"===== Unintegrated system: {n} input relations =====\n")
        f.write(f"Import time: {total_import_time} seconds\n")
        f.write(f"Join time: {total_join_time} seconds\n")
        f.write(f"Training time: {training_time} seconds\n\n")

def main():
    # if os.path.exists('test/quackml/benchmark_results.txt'):
    #     os.remove('test/quackml/benchmark_results.txt')

    if os.path.exists('test/quackml/factorised_test_results.txt'):
        os.remove('test/quackml/factorised_test_results.txt')

    # test_linear_regression_benchmark('test/quackml/datasets/test_8.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_9.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_10.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_11.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_12.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_13.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_14.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_15.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_16.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_17.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_18.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_19.tsv')
    # test_linear_regression_benchmark('test/quackml/datasets/test_20.tsv')

    test_cartesian_product('test/quackml/datasets/factorised', 1)
    test_cartesian_product('test/quackml/datasets/factorised', 2)
    test_cartesian_product('test/quackml/datasets/factorised', 3)
    test_cartesian_product('test/quackml/datasets/factorised', 4)
    test_cartesian_product('test/quackml/datasets/factorised', 5)
    test_cartesian_product('test/quackml/datasets/factorised', 6)
    test_cartesian_product('test/quackml/datasets/factorised', 7)
    test_cartesian_product('test/quackml/datasets/factorised', 8)
    test_cartesian_product('test/quackml/datasets/factorised', 9)
    test_cartesian_product('test/quackml/datasets/factorised', 10)

if __name__ == "__main__":
    main()