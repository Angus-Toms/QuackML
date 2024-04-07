// linear_reg_ring.hpp

#ifndef LINEAR_REG_RING_TYPE_HPP
#define LINEAR_REG_RING_TYPE_HPP

#pragma once
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/vector.hpp"

#include "duckdb/common/types/value.hpp"

namespace quackml {

struct LinRegRing {
    double count;
    std::vector<double> sums;
    std::vector<std::vector<double>> covar;

    // Constructor from feature observations, used by to_ring
    LinRegRing(std::vector<duckdb::vector<duckdb::Value>> &observations);
    // Constructor used in linear_reg_ring
    // Values loaded from list value
    LinRegRing(double count, std::vector<double> sums, std::vector<std::vector<double>> covar);
    // Copy constructor 
    LinRegRing(const LinRegRing &other);

    // Operators
    LinRegRing &operator+(const LinRegRing &other);
    LinRegRing &operator*(const LinRegRing &other);

    // Helpers 
    void padUpper(idx_t inc);
    void padLower(idx_t inc);
    void print();
};

} // namespace quackml

#endif // LINEAR_REG_RING_TYPE_HPP
