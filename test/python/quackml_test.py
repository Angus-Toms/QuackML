import duckdb
import os
import pytest
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import time

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
