// UDAF to perform 1D regularised linear regression 
// Build with new constructor

#include "functions/linear_reg.hpp"

// Testing:
#include <iostream>
using std::cout;
using std::endl;

namespace quackml {

float getGradient(idx_t n, float Sigma, float C, float theta) {
    float lambda = 0.0; // Set regularisation to 0 for now
    return (1.0 / n) * ((Sigma * theta - C) + (lambda * theta));
}

struct LinearRegState {
    idx_t count;
    double theta;
    double Sigma;
    double C;
};

struct LinearRegFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.count = 0;
        state.theta = 1.0;
        state.Sigma = 0.0;
        state.C = 0.0;
    }

    template <class A_TYPE, class B_TYPE, class STATE, class OP>
    static void Operation(STATE &state, const A_TYPE &feature, const B_TYPE &label, duckdb::AggregateBinaryInput &idata) {
        state.count++;
        state.Sigma += feature * feature;
        state.C += feature * label;
    }

    template <class STATE, class OP>
    static void Combine(const STATE &source, STATE &target, duckdb::AggregateInputData &data) {
        target.count += source.count;
        target.Sigma += source.Sigma;
        target.C += source.C;
    }

    static bool IgnoreNull() {
        return true;
    }

    template <class T, class STATE>
    static void Finalize(STATE &state, T &target, duckdb::AggregateFinalizeData &finalize_data) {
        
        if (state.count == 0) {
            finalize_data.ReturnNull();
        }
        
        // Gradient descent 
        float alpha = 0.01;
        for (idx_t i = 0; i < 1000; i++) {
            auto gradient = getGradient(state.count, state.Sigma, state.C, state.theta);
            state.theta -= alpha * gradient;
        }
        target = state.theta;
    }

};

duckdb::AggregateFunction GetLinearRegFunction() {
    return duckdb::AggregateFunction::BinaryAggregate<LinearRegState, double, double, double, LinearRegFunction>(duckdb::LogicalType::DOUBLE, duckdb::LogicalType::DOUBLE, duckdb::LogicalType::DOUBLE);
} 

void LinearRegression::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet linear_reg("linear_regression");
    linear_reg.AddFunction(GetLinearRegFunction());
    duckdb::CreateAggregateFunctionInfo info(linear_reg);
    catalog.CreateFunction(*conn.context, info);
}

}// namespace quackml