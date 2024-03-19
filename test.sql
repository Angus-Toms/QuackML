CREATE TABLE tbl_1 (feature INT, id CHAR);
INSERT INTO tbl_1 VALUES (1, 'a'), (2, 'b'), (3, 'a');
CREATE TABLE tbl_2 (feature INT, id CHAR, key CHAR);
INSERT INTO tbl_2 VALUES (4, 'a', 'x'), (5, 'b', 'x'), (6, 'a', 'z');
CREATE TABLE tbl_3 (feature INT, key CHAR);
INSERT INTO tbl_3 VALUES (7, 'x'), (8, 'y'), (9, 'z');

-- Inspect join 
SELECT * FROM tbl_1 JOIN tbl_2 ON tbl_1.id = tbl_2.id;

-- Get ring from join 
SELECT to_ring([tbl_1.feature, tbl_2.feature]) FROM tbl_1 JOIN tbl_2 ON tbl_1.id = tbl_2.id;

-- Perform linear regression on cartesian product
SELECT linear_regression_ring([t1.ring, t2.rings], 0) FROM
(SELECT to_ring([feature]) ring FROM tbl_1) t1,
(SELECT to_ring([feature]) ring FROM tbl_2) t2;

-- Linear regression over join on id and key columns
SELECT linear_regression_ring([t1.ring, t2.ring, t3.ring], 1) FROM 
(SELECT id, to_ring([feature]) ring FROM tbl_1 GROUP BY id) t1,
(SELECT id, key, to_ring([feature]) ring FROM tbl_2 GROUP BY id, key) t2,
(SELECT key, to_ring([feature]) ring FROM tbl_3 GROUP BY key) t3
WHERE t1.id = t2.id AND t2.key = t3.key;

-- Linear regression over join on id, grouping by key
SELECT t2.key, linear_regression_ring([t1.ring, t2.ring], 0) FROM 
(SELECT id, to_ring([feature]) ring FROM tbl_1 GROUP BY id) t1,
(SELECT key, id, to_ring([feature]) ring FROM tbl_2 GROUP BY key, id) t2
WHERE t1.id = t2.id GROUP BY t2.key;