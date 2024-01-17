# QuackML

This repository is based on https://github.com/duckdb/extension-template, check it out if you want to build and ship your own DuckDB extension.

---

QuackML aims to implement various ML tasks including regularised linear regression, Naive Bayes classifiers, and mutual information following principles of in-database machine learning. Under this approach, ML tasks are decomposed into batches of aggregate queries that are then evualuated within the database system. This project extension is a work in progress.