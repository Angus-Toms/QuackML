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
    std::cout << " <=========== Running QuackML tests ===========>\n";

    Connection con(db);

    // MUNGO TODO: Linear regression tests 
    con.Query("CREATE TABLE t (features INTEGER[], label INTEGER);");
    con.Query("INSERT INTO t VALUES ([1, 2], -4), ([3, 1], 3), ([4, 5], -7);"); // y = 2x_1 - 3x_2
    con.Query("SELECT * FROM t;")->Print();
    con.Query("SELECT linear_regression(features, label, 0.01, 0, 500) AS linear_regression FROM t")->Print();

    // MUNGO TODO: Linear regression ring tests

    std::cout << " <========== QuackML tests complete ==========>\n";
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

    #ifdef QUACKML_TESTS
        run_quackml_tests(db);
    #endif
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
