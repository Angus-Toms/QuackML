#define DUCKDB_EXTENSION_MAIN

#include "quackml_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include "functions/sum.hpp"
#include "functions/sum_count.hpp"
#include "functions/linear_reg.hpp"
#include "functions/to_ring.hpp"
#include "functions/linear_reg_ring.hpp"
#include "types/linear_reg_ring.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

#include <iostream>

namespace duckdb {

void run_quackml_tests(DuckDB &db) {
    std::cout << "<=========== Running QuackML tests ===========>\n";

    Connection con(db);

    // Linear regression tests 
    // con.Query("CREATE TABLE csv AS SELECT * FROM read_csv('test/quackml/test_1000000.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    // auto start_time = std::chrono::high_resolution_clock::now();
    // con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv;")->Print();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::cout << "Linear regression time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

    con.Query("CREATE TABLE a (xa INTEGER[]);");
    con.Query("INSERT INTO a VALUES ([1]), ([3]), ([-1]);");
    con.Query("CREATE TABLE b (xb INTEGER[]);");
    con.Query("INSERT INTO b VALUES ([7, 7]), ([5, 5]), ([3, 3]);");
    con.Query("SELECT linear_regression_ring([a_ring, b_ring], 0) FROM (SELECT to_ring(xa) a_ring FROM a), (SELECT to_ring(xb) b_ring FROM b);")->Print();

    // MUNGO TODO: Linear regression ring tests

    std::cout << "<========== QuackML tests complete ==========>\n";
}


void QuackmlExtension::Load(DuckDB &db) {
	Connection con(db);
    con.BeginTransaction();
    auto &catalog = Catalog::GetSystemCatalog(*con.context);

    quackml::Sum::RegisterFunction(con, catalog);
    quackml::SumCount::RegisterFunction(con, catalog);
    quackml::LinearRegression::RegisterFunction(con, catalog);
    quackml::ToRing::RegisterFunction(con, catalog);
    quackml::LinearRegressionRing::RegisterFunction(con, catalog);

    con.Commit();

    run_quackml_tests(db);
}

std::string QuackmlExtension::Name() {
	return "QuackML";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void quackml_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::QuackmlExtension>();
}

DUCKDB_EXTENSION_API const char *quackml_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
