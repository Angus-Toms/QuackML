CREATE TABLE t (features DOUBLE[], label DOUBLE);
INSERT INTO t values ([1, 2, 3], 9), ([4, 1, 2], 20);
SELECT linear_regression(features, label, 0.01, 0.0, 1000) FROM t;