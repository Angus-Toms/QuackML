// linear_reg.hpp

#ifndef LINEAR_REG_HPP
#define LINEAR_REG_HPP

#pragma once 
// TODO: check if all includes are necessary
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/vector.hpp"

namespace quackml {

struct LinearRegression {
    static void RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog);
};

} // namespace quackml

#endif // LINEAR_REG_HPP