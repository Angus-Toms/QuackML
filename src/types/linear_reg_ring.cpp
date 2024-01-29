#include "types/linear_reg_ring.hpp"

#include <iostream>
#include <stdexcept>    

// ASSUMPTIONS:
// We assume that each input relation contains labels
// All relation feature-lists are disjoint 
// Relations are joined on a non-feature non-label column (e.g. id)

namespace quackml {

LinearRegressionRingElement::LinearRegressionRingElement() {
    count = duckdb::Value::DOUBLE(0);
}
// Create ring with dxd covariance matrix of 0s
LinearRegressionRingElement::LinearRegressionRingElement(idx_t d) {
    this->d = d;
    count = duckdb::Value::DOUBLE(0);

    auto sums_vector = duckdb::vector<duckdb::Value>(d);
    auto covar_vector = duckdb::vector<duckdb::Value>(d);
    for (idx_t i = 0; i < d; i++) {
        sums_vector[i] = duckdb::Value::DOUBLE(0);
        auto row = duckdb::vector<duckdb::Value>(d);
        for (idx_t j = 0; j < d; j++) {
            row[j] = duckdb::Value::DOUBLE(0);
        }
        covar_vector[i] = duckdb::Value::LIST(row);
    }
    sums = duckdb::Value::LIST(sums_vector);
    covar = duckdb::Value::LIST(covar_vector);
}
LinearRegressionRingElement::LinearRegressionRingElement(duckdb::vector<duckdb::Value> features) {
    d = features.size();
    
    count = duckdb::Value::DOUBLE(1);

    sums = duckdb::Value::LIST(features);

    auto covar_vector = duckdb::vector<duckdb::Value>(d);
    for (idx_t i = 0; i < d; i++) {
        auto row = duckdb::vector<duckdb::Value>(d, duckdb::Value::DOUBLE(0));
        auto feature_value = features[i].GetValue<double>();
        row[i] = duckdb::Value::DOUBLE(feature_value * feature_value);
        covar_vector[i] = duckdb::Value::LIST(row);
    }
    covar = duckdb::Value::LIST(covar_vector);
}

LinearRegressionRingElement::~LinearRegressionRingElement() {}

LinearRegressionRingElement LinearRegressionRingElement::operator+(const LinearRegressionRingElement &other) {
    if (d != other.d) {
        throw std::invalid_argument("Cannot add LinearRegressionRingElements with different dimensions");
    }
    LinearRegressionRingElement result(d);
    result.count = duckdb::Value::DOUBLE(count.GetValue<double>() + other.count.GetValue<double>());

    // Sums 
    duckdb::vector<duckdb::Value> sums_vector(d);
    auto sum_children = duckdb::ListValue::GetChildren(sums);
    auto other_sum_children = duckdb::ListValue::GetChildren(other.sums);
    for (idx_t i = 0; i < d; i++) {
        sums_vector[i] = duckdb::Value::DOUBLE(sum_children[i].GetValue<double>() + other_sum_children[i].GetValue<double>());
    }
    result.sums = duckdb::Value::LIST(sums_vector);

    // Covariance matrix
    duckdb::vector<duckdb::Value> covar_vector(d);
    auto covar_children = duckdb::ListValue::GetChildren(covar);
    auto other_covar_children = duckdb::ListValue::GetChildren(other.covar);
    for (idx_t i = 0; i < d; i++) {
        auto row = duckdb::vector<duckdb::Value>(d);
        auto row_children = duckdb::ListValue::GetChildren(covar_children[i]);
        auto other_row_children = duckdb::ListValue::GetChildren(other_covar_children[i]);
        for (idx_t j = 0; j < d; j++) {
            row[j] = duckdb::Value::DOUBLE(row_children[j].GetValue<double>() + other_row_children[j].GetValue<double>());
        }
        covar_vector[i] = duckdb::Value::LIST(row);
    }
    result.covar = duckdb::Value::LIST(covar_vector);

    return result;
}

    // LinearRegressionRingElement operator+(const LinearRegressionRingElement &other) {
    //     if (d != other.d) {
    //         throw std::invalid_argument("Cannot add LinearRegressionRingElements with different dimensions");
    //     }

    //     LinearRegressionRingElement result;

    //     // Counts
    //     result.count = count + other.count;

    //     // Sums and covariance matrix
    //     for (idx_t i = 0; i < d; i++) {
    //         result.sums[i] = sums[i] + other.sums[i];
    //         for (idx_t j = 0; j < d; j++) {
    //             result.covar[i][j] = covar[i][j] + other.covar[i][j];
    //         }
    //     }
    //     return result;
    // }

    // // HMMMMMM: return to this.
    // LinearRegressionRingElement operator*(const LinearRegressionRingElement &other) {
    //     // Create covar matrix of combined features
    //     LinearRegressionRingElement result(d + other.d);

    //     // Counts
    //     result.count = count * other.count;

    //     // Sums 
    //     for (idx_t i = 0; i < d; i++) {
    //         result.sums[i] = other.sums[i] * count + sums[i] * other.count;
    //     }

    //     // Covariance matrix 
    //     // See: https://www.youtube.com/watch?v=W6GrRmqvLyc&list=PL7dGmiBS0RtfVhhedAG7OzrL6nygjgGyn&index=3
    //     for (idx_t i = 0; i < d; i++) {
    //         for (idx_t j = 0; j < d; j++) {
    //             result.covar[i][j] = (other.count * covar[i][j]) + (count * other.covar[i][j]) + (sums[i] * other.sums[j]) + (other.sums[i] * sums[j]);
    //         }
    //     }
    //     return result;
    // }

    // int getD() {
    //     return d;
    // }

    // double getCount() {
    //     return count;
    // }

    // std::vector<double>* getSums() {
    //     return &sums;
    // }

    // std::vector<std::vector<double>>* getCovar() {
    //     return &covar;
    // }

} // namespace quackml