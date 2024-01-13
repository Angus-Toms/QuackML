// sum_count.hpp 

#ifndef SUM_COUNT_HPP
#define SUM_COUNT_HPP

#pragma once
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/vector.hpp"

#include <map>

namespace quackml {

struct SumCountState {
    idx_t sum;
    double count;
};

struct SumCountFunction;

struct SumCount {
    static void RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog);
};

} // namespace quackml

#endif // SUM_COUNT_HPP