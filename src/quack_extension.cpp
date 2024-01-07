#define DUCKDB_EXTENSION_MAIN

#include "quack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include "functions/sum.cpp" // This is bad, need header files

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

void QuackExtension::Load(DuckDB &db) {
	Connection con(db);
    con.BeginTransaction();
    auto &catalog = Catalog::GetSystemCatalog(*con.context);
    ml::RegisterSumFunction(con, catalog);
    con.Commit();
}
std::string QuackExtension::Name() {
	return "quack";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void quack_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::QuackExtension>();
}

DUCKDB_EXTENSION_API const char *quack_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
