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

namespace quackml {

const duckdb::LogicalType MatrixType = duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE));
const duckdb::LogicalType LinearRegressionRingType = duckdb::LogicalType::LIST(MatrixType);

struct LinearRegressionRingElement {
private:
    idx_t d;
    duckdb::Value count;    // scalar
    duckdb::Value sums;     // column vector 
    duckdb::Value c;        // column vector
    duckdb::Value covar;    // matrix

public:
    LinearRegressionRingElement();
    LinearRegressionRingElement(idx_t d, bool additive_identity);
    LinearRegressionRingElement(duckdb::vector<duckdb::Value> features);
    LinearRegressionRingElement(duckdb::Value ring);
    ~LinearRegressionRingElement();

    // Operands
    LinearRegressionRingElement operator+(const LinearRegressionRingElement &other);
    LinearRegressionRingElement operator*(const LinearRegressionRingElement &other);

    void set_c(duckdb::Value c_arg) { c = c_arg; }

    idx_t get_d() { return d; }
    duckdb::Value get_count() { return count; }
    duckdb::Value get_sums() { return sums; }
    duckdb::Value get_covar() { return covar; }
    duckdb::Value get_c() { return c; }

    // Wrapped accessors, make count and sums matrices to allow for uniform list type
    // covar is already a matrix so no new accessor needed
    duckdb::Value get_count_wrapped() { return duckdb::Value::LIST({duckdb::Value::LIST({count})}); }
    duckdb::Value get_sums_wrapped() { return duckdb::Value::LIST({sums}); }

    // Padding - add rows and columns of 0s to top left or bottom right of covariance matrix
    void pad_upper(idx_t d_inc);
    void pad_lower(idx_t d_inc);

    void Print();
};

} // namespace quackml

#endif // LINEAR_REG_RING_TYPE_HPP