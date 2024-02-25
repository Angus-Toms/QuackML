# QuackML

QuackML is a [DuckDB](https://www.duckdb.org) extension and aims to implement various ML tasks including regularised linear regression, Naive Bayes classifiers, and mutual information following principles of in-database machine learning. Under this approach, ML tasks are decomposed into batches of aggregate queries that are then evualuated within the database system. This extension is a work in progress.  

**Disclaimer**: This extension is in no way affiliated with the [DuckDB Foundation](https://duckdb.org/foundation/) or [DuckDB Labs](https://duckdblabs.com/). Therefore, any binaries produced and distributed of this extension are unsigned.

## Design 
This repository is based on https://github.com/duckdb/extension-template, check it out if you want to build and ship your own DuckDB extension.

## Usage 
To make this project first run 
```bash
make
```
from the project directory and then build with the unsigned flag 
```bash 
./build/release/duckdb -unsigned
```
This laucnhes the DuckDB CLI tool, then load the extension by running:
```SQL 
LOAD 'extension/quackml/quackml.duckdb_extension';
```
and you are then free to use QuackML's new functions:
```SQL
CREATE TABLE test (features INTEGER[], label INTEGER);
INSERT INTO test VALUES ([1, 2], 4), ([-1, 0], -2), ([3, 5], 11);
SELECT linear_regression(features, label, 0.1, 0, 1000) as linear_regression FROM test;

┌───────────────────────────────┐
│       linear_regression       │
│           double[]            │
├───────────────────────────────┤
│ [2.0000000000, 1.0000000000]  │
└───────────────────────────────┘
```