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
    // con.Query("CREATE TABLE csv AS SELECT * FROM read_csv('test/quackml/test_100.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    // auto start_time = std::chrono::high_resolution_clock::now();
    // con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv;")->Print();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::cout << "Linear regression time (100 obs): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

    con.Query("CREATE TABLE csv_1 AS SELECT * FROM read_csv('test/quackml/test_100.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    auto start_time = std::chrono::high_resolution_clock::now();
    con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv_1;")->Print();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Linear regression time (100 obs): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

    con.Query("CREATE TABLE csv_2 AS SELECT * FROM read_csv('test/quackml/test_1000.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    start_time = std::chrono::high_resolution_clock::now();
    con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv_2;")->Print();
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Linear regression time (1000 obs): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

    con.Query("CREATE TABLE csv_3 AS SELECT * FROM read_csv('test/quackml/test_10000.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    start_time = std::chrono::high_resolution_clock::now();
    con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv_3;")->Print();
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Linear regression time (10000 obs): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

    con.Query("CREATE TABLE csv_4 AS SELECT * FROM read_csv('test/quackml/test_100000.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    start_time = std::chrono::high_resolution_clock::now();
    con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv_4;")->Print();
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Linear regression time (100000 obs): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

    con.Query("CREATE TABLE csv_5 AS SELECT * FROM read_csv('test/quackml/test_1000000.tsv', header = TRUE, delim = '\t', columns = {'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    start_time = std::chrono::high_resolution_clock::now();
    con.Query("SELECT linear_regression(features, label, 0) as linear_regression FROM csv_5;")->Print();
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Linear regression time (1000000 obs): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms\n";

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
