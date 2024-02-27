#include "types/linear_reg_ring.hpp"

#include <iostream>
#include <stdexcept>    

// ASSUMPTIONS:
// We assume that each input relation contains labels
// All relation feature-lists are disjoint 
// Relations are joined on a non-feature non-label column (e.g. id)

namespace quackml {

LinearRegressionRingElement::LinearRegressionRingElement() {}
// Create a d-dimensional ring element, assert additive identity to make 
// element the additive identity, multiplicative identity otherwise
LinearRegressionRingElement::LinearRegressionRingElement(idx_t d_arg, bool additive_identity) {
    d = d_arg;

    count = additive_identity ? duckdb::Value::DOUBLE(0) : duckdb::Value::DOUBLE(1);
    auto sum_vector = duckdb::vector<duckdb::Value>(d, duckdb::Value::DOUBLE(0));

    auto covar_vector = duckdb::vector<duckdb::Value>(d);
    for (idx_t i = 0; i < d; i++) {
        auto row = duckdb::vector<duckdb::Value>(d, duckdb::Value::DOUBLE(0));
        covar_vector[i] = duckdb::Value::LIST(row);
    }
    sums = duckdb::Value::LIST(sum_vector);
    covar = duckdb::Value::LIST(covar_vector);
}


// Constructor used in to_ring function
LinearRegressionRingElement::LinearRegressionRingElement(duckdb::vector<duckdb::Value> features) {
    d = features.size();
    count = duckdb::Value::DOUBLE(1);
    sums = duckdb::Value::LIST(features);

    // auto covar_vector = duckdb::vector<duckdb::Value>(d);
    // for (idx_t i = 0; i < d; i++) {
    //     auto row = duckdb::vector<duckdb::Value>(d, duckdb::Value::DOUBLE(0));
    //     auto feature_value = features[i].GetValue<double>();
    //     row[i] = duckdb::Value::DOUBLE(feature_value * feature_value);
    //     covar_vector[i] = duckdb::Value::LIST(row);
    // }

    auto covar_vector = duckdb::vector<duckdb::Value>(d);
    for (idx_t i = 0; i < d; i++) {
        auto row = duckdb::vector<duckdb::Value>(d, duckdb::Value::DOUBLE(0));
        for (idx_t j = 0; j < d; j++) {
            row[j] = duckdb::Value::DOUBLE(features[i].GetValue<double>() * features[j].GetValue<double>());
        }
        covar_vector[i] = duckdb::Value::LIST(row);
    }
    covar = duckdb::Value::LIST(covar_vector);
}
// Constructor used in linear_regression_ring function. Values are loaded from to_ring 
// subqueries so returned as 3D duckdb::LIST 
LinearRegressionRingElement::LinearRegressionRingElement(duckdb::Value ring) {
    auto ring_children = duckdb::ListValue::GetChildren(ring);

    // Unwrap "matrix-ified" count and sums
    count = duckdb::ListValue::GetChildren(duckdb::ListValue::GetChildren(ring_children[0])[0])[0];
    sums = duckdb::ListValue::GetChildren(ring_children[1])[0];
    covar = ring_children[2];
    d = duckdb::ListValue::GetChildren(sums).size();
}

LinearRegressionRingElement::~LinearRegressionRingElement() {}

LinearRegressionRingElement LinearRegressionRingElement::operator+(const LinearRegressionRingElement &other) {
    if (d != other.d) {
        throw std::invalid_argument("Cannot add LinearRegressionRingElements with different dimensions");
    }
    LinearRegressionRingElement result(d, true);
    result.count = duckdb::Value::DOUBLE(count.GetValue<double>() + other.count.GetValue<double>());

    // Sums 
    duckdb::vector<duckdb::Value> sums_vector(d);
    duckdb::vector<duckdb::Value> covar_vector(d);

    auto sum_children = duckdb::ListValue::GetChildren(sums);
    auto other_sum_children = duckdb::ListValue::GetChildren(other.sums);
    auto covar_children = duckdb::ListValue::GetChildren(covar);
    auto other_covar_children = duckdb::ListValue::GetChildren(other.covar);

    for (idx_t i = 0; i < d; i++) {
        sums_vector[i] = duckdb::Value::DOUBLE(sum_children[i].GetValue<double>() + other_sum_children[i].GetValue<double>());

        auto row = duckdb::vector<duckdb::Value>(d);
        auto row_children = duckdb::ListValue::GetChildren(covar_children[i]);
        auto other_row_children = duckdb::ListValue::GetChildren(other_covar_children[i]);
        for (idx_t j = 0; j < d; j++) {
            row[j] = duckdb::Value::DOUBLE(row_children[j].GetValue<double>() + other_row_children[j].GetValue<double>());
        }
        covar_vector[i] = duckdb::Value::LIST(row);
    }

    result.sums = duckdb::Value::LIST(sums_vector);
    result.covar = duckdb::Value::LIST(covar_vector);

    return result;
}

LinearRegressionRingElement LinearRegressionRingElement::operator*(const LinearRegressionRingElement &other) {
    // MUNGO TODO: Add bias term at this stage? 
    // Or add during finalize routine?
    if (d != other.d) {
        throw std::invalid_argument("Cannot multiply LinearRegressionRingElements with different dimensions");
    }
    LinearRegressionRingElement result(d, true);
    
    // Count
    double count_val = count.GetValue<double>();
    double other_count_val = other.count.GetValue<double>();
    result.count = duckdb::Value::DOUBLE(count_val * other_count_val);

    // Sums and Covariance
    duckdb::vector<duckdb::Value> sums_vector(d);
    duckdb::vector<duckdb::Value> covar_vector(d);
    auto sum_children = duckdb::ListValue::GetChildren(sums);
    auto other_sum_children = duckdb::ListValue::GetChildren(other.sums);
    auto covar_children = duckdb::ListValue::GetChildren(covar);
    auto other_covar_children = duckdb::ListValue::GetChildren(other.covar);

    for (idx_t i = 0; i < d; i++) {
        sums_vector[i] = duckdb::Value::DOUBLE(
            sum_children[i].GetValue<double>() * other_count_val +
            other_sum_children[i].GetValue<double>() * count_val
        );

        auto row = duckdb::vector<duckdb::Value>(d);
        auto row_children = duckdb::ListValue::GetChildren(covar_children[i]);
        auto other_row_children = duckdb::ListValue::GetChildren(other_covar_children[i]);
        for (idx_t j = 0; j < d; j++) {
            row[j] = duckdb::Value::DOUBLE(
                other_count_val * row_children[j].GetValue<double>() +
                count_val * other_row_children[j].GetValue<double>() +
                sum_children[i].GetValue<double>() * other_sum_children[j].GetValue<double>() +
                other_sum_children[i].GetValue<double>() * sum_children[j].GetValue<double>()
            );
        }
        covar_vector[i] = duckdb::Value::LIST(row);
    }
    result.sums = duckdb::Value::LIST(sums_vector);
    result.covar = duckdb::Value::LIST(covar_vector);

    return result;
}

void LinearRegressionRingElement::pad_upper(idx_t d_inc) {
    // Add 0s to top of sum vector 
    auto sum_vector = duckdb::vector<duckdb::Value>(d_inc, duckdb::Value::DOUBLE(0));
    auto sum_children = duckdb::ListValue::GetChildren(sums);
    for (auto &sum_child : sum_children) {
        sum_vector.push_back(sum_child);
    }

    // Add d_inc rows to top of covar, each with (d+d_inc) 0s
    auto covar_vector = duckdb::vector<duckdb::Value>(d_inc, duckdb::Value::LIST(duckdb::vector<duckdb::Value>(d + d_inc, duckdb::Value::DOUBLE(0))));
    auto covar_children = duckdb::ListValue::GetChildren(covar);

    // Add d_inc 0s to left of each row 
    for (auto &covar_row : covar_children) {
        auto new_row = duckdb::vector<duckdb::Value>(d_inc, duckdb::Value::DOUBLE(0));
        for (auto &element : duckdb::ListValue::GetChildren(covar_row)) {
            new_row.push_back(element);
        }
        covar_vector.push_back(duckdb::Value::LIST(new_row));
    }
    sums = duckdb::Value::LIST(sum_vector);
    covar = duckdb::Value::LIST(covar_vector);
    d += d_inc;
}

void LinearRegressionRingElement::pad_lower(idx_t d_inc) {
    //  Add 0s to bottom of sum vector 
    auto sum_vector = duckdb::ListValue::GetChildren(sums);
    for (idx_t i = 0; i < d_inc; i++) {
        sum_vector.push_back(duckdb::Value::DOUBLE(0));
    }

    // Extend first d rows by d_inc
    auto covar_vector = duckdb::ListValue::GetChildren(covar);
    for (idx_t i = 0; i < d; i++) {
        auto row = duckdb::ListValue::GetChildren(covar_vector[i]);
        for (idx_t j = 0; j < d_inc; j++) {
            row.push_back(duckdb::Value::DOUBLE(0));
        }
        covar_vector[i] = duckdb::Value::LIST(row);
    }

    // Add d_inc rows to bottom, each with (d+d_inc) 0s
    for (idx_t i = 0; i < d_inc; i++) {
        auto row = duckdb::vector<duckdb::Value>(d + d_inc, duckdb::Value::DOUBLE(0));
        covar_vector.push_back(duckdb::Value::LIST(row));
    }
    
    sums = duckdb::Value::LIST(sum_vector);
    covar = duckdb::Value::LIST(covar_vector);
    d += d_inc;
}

void LinearRegressionRingElement::Print() {
    std::cout << "Count: " << count.GetValue<double>() << std::endl;
    std::cout << "Sums: ";
    auto sum_children = duckdb::ListValue::GetChildren(sums);
    for (idx_t i = 0; i < d; i++) {
        std::cout << sum_children[i].GetValue<double>() << " ";
    }
    std::cout << std::endl;
    std::cout << "Covar: " << std::endl;
    auto covar_children = duckdb::ListValue::GetChildren(covar);
    for (idx_t i = 0; i < d; i++) {
        auto row_children = duckdb::ListValue::GetChildren(covar_children[i]);
        for (idx_t j = 0; j < d; j++) {
            std::cout << row_children[j].GetValue<double>() << " ";
        }
        std::cout << std::endl;
    }
}



} // namespace quackml