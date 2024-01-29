// linear_reg_ring.hpp

#ifndef LINEAR_REG_RING_HPP
#define LINEAR_REG_RING_HPP

#pragma once
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/vector.hpp"

namespace quackml {

struct LinearRegressionRingElement {
private:
    idx_t d;
    duckdb::Value count;
    duckdb::Value sums;
    duckdb::Value covar;    

public:
    LinearRegressionRingElement();
    LinearRegressionRingElement(idx_t d);
    LinearRegressionRingElement(duckdb::vector<duckdb::Value> features);
    ~LinearRegressionRingElement();

    // Operands
    LinearRegressionRingElement operator+(const LinearRegressionRingElement &other);
    LinearRegressionRingElement operator*(const LinearRegressionRingElement &other);

    idx_t get_d() { return d; }
    duckdb::Value get_count() { return count; }
    duckdb::Value get_sums() { return sums; }
    duckdb::Value get_covar() { return covar; }

    void Print();
};

} // namespace quackml

#endif