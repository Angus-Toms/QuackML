CREATE TABLE my_table (feature DOUBLE, label DOUBLE);
INSERT INTO my_table VALUES (2, -9), (4, -18), (-3, 13.5);
SELECT linear_regression_calls(feature, label, 0.01, 0.0, 1000) FROM my_table;