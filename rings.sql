/* FEATURES 1 */
CREATE TABLE a (i INTEGER[]);
INSERT INTO a VALUES ([1, 2]), ([3, 4]), ([5, 6]);

/* FEATURES 2 */ 
CREATE TABLE b (j INTEGER[]);
INSERT INTO b VALUES ([4, 3, 2]), ([9, 1, 5]), ([7, 0, -1]);

/* Linear regression with rings */
SELECT linear_regression_ring([i, j], 5, 0.01, 0.0, 1000)
FROM (
    SELECT to_ring(i) i
    FROM a
),
(
    SELECT to_ring(j) j
    FROM b
);