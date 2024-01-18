CREATE TABLE t (feature DOUBLE, label DOUBLE);
INSERT INTO t VALUES (2, 6), (4, 12), (-3, -9), (5, 15), (1, 3), (0, 0), (10, 30), (7, 21), (8, 24), (9, 27);
SELECT linear_regression_query(feature, label, 0.01, 0.0, 100) FROM t;