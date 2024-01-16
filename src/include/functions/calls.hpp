// sum_calls.hpp

#ifndef SUM_CALLS_HPP
#define SUM_CALLS_HPP

#pragma once
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

namespace quackml {

struct SumCallsState {
    double sum;
};

struct SumCallsOperation;

struct SumCalls {
    static void RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog);
};

} // namespace quackml

#endif // SUM_HPP