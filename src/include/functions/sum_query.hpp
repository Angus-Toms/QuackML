// sum_query.hpp

#ifndef SUM_QUERY_HPP
#define SUM_QUERY_HPP

#pragma once
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

namespace quackml {

struct SumQueryState {
    double sum;
};

struct SumQueryOperation;

struct SumQuery {
    static void RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog);
};

} // namespace quackml

#endif // SUM_QUERY_HPP