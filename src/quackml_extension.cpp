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
#include "functions/calls.hpp"
#include "functions/linear_reg_calls.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

void QuackmlExtension::Load(DuckDB &db) {
	Connection con(db);
    con.BeginTransaction();
    auto &catalog = Catalog::GetSystemCatalog(*con.context);

    quackml::Sum::RegisterFunction(con, catalog);
    quackml::SumCount::RegisterFunction(con, catalog);
    quackml::LinearRegression::RegisterFunction(con, catalog);
    quackml::SumCalls::RegisterFunction(con, catalog);
    quackml::LinearRegressionCalls::RegisterFunction(con, catalog);

    con.Commit();
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
