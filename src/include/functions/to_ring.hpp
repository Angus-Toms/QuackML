// to_ring.hpp 

#ifndef TO_RING_HPP
#define TO_RING_HPP

#pragma once
#include "types/linear_reg_ring.hpp"

#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/vector.hpp"

#include <iostream>

namespace quackml {

struct ToRing {
    static void RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog);
};

} // namespace quackml

#endif