// UDAF to perform 1D regularised linear regression 
// Build with new constructor

#include "functions/linear_reg.hpp"

// Testing:
#include <iostream>
#include <vector>
#include <typeinfo> 
using std::cout;
using std::endl;
using std::vector;

namespace quackml {

float getGradient(idx_t n, float Sigma, float C, float theta, float lambda) {
    return (1.0 / n) * ((Sigma * theta - C) + (lambda * theta));
}

// struct LinearRegState {
//     idx_t count;
//     double theta;
//     double Sigma;
//     double C;
// };

// struct LinearRegFunction {
//     template <class STATE>
//     static void Initialize(STATE &state) {
//         state.count = 0;
//         state.theta = 1.0;
//         state.Sigma = 0.0;
//         state.C = 0.0;
//     }

//     static bool IgnoreNull() {
//         return true;
//     }

//     template <class T, class STATE>
//     static void Finalize(STATE &state, T &target, duckdb::AggregateFinalizeData &finalize_data) {
        
//         if (state.count == 0) {
//             finalize_data.ReturnNull();
//         }
        
//         // Gradient descent 
//         float alpha = 0.01;
//         for (idx_t i = 0; i < 1000; i++) {
//             auto gradient = getGradient(state.count, state.Sigma, state.C, state.theta);
//             state.theta -= alpha * gradient;
//         }
//         target = state.theta;
//     }

// };

// duckdb::AggregateFunction GetLinearRegFunction() {
//     return duckdb::AggregateFunction::BinaryAggregate<LinearRegState, double, double, double, LinearRegFunction>(duckdb::LogicalType::DOUBLE, duckdb::LogicalType::DOUBLE, duckdb::LogicalType::DOUBLE);
// } 

// void LinearRegression::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
//     duckdb::AggregateFunctionSet linear_reg("linear_regression");
//     linear_reg.AddFunction(GetLinearRegFunction());
//     duckdb::CreateAggregateFunctionInfo info(linear_reg);
//     catalog.CreateFunction(*conn.context, info);
// }

struct LinearRegressionState {
    double theta;

    idx_t count;
    double Sigma;
    double C;

    double alpha;
    double lambda;
    int iterations;
};


struct LinearRegressionFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.theta = 1.0;

        state.count = 0;
        state.Sigma = 0.0;
        state.C = 0.0;

        state.alpha = 100;
        state.lambda = 0.0;
        state.iterations = 1000;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        return;
    }

    static bool IgnoreNull() {
        return true;
    }
};

static void LinearRegressionUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &feature = inputs[0];
    auto &label = inputs[1];
    auto &alpha = inputs[2];
    auto &lambda = inputs[3];
    auto &iterations = inputs[4];

    duckdb::UnifiedVectorFormat feature_data;
    duckdb::UnifiedVectorFormat label_data;
    duckdb::UnifiedVectorFormat alpha_data;
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat iterations_data;

    feature.ToUnifiedFormat(count, feature_data);
    label.ToUnifiedFormat(count, label_data);
    alpha.ToUnifiedFormat(count, alpha_data);
    lambda.ToUnifiedFormat(count, lambda_data);
    iterations.ToUnifiedFormat(count, iterations_data);

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        if (feature_data.validity.RowIsValid(feature_data.sel->get_index(i)) && label_data.validity.RowIsValid(label_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];

            auto feature_value = feature.GetValue(i);
            auto feature_children = duckdb::ListValue::GetChildren(feature_value);
            std::cout << "feature: " << feature_children[0] << "\n";
            // TODO: Start here
            auto label_value = duckdb::UnifiedVectorFormat::GetData<double>(label_data)[label_data.sel->get_index(i)];
            state.alpha = duckdb::UnifiedVectorFormat::GetData<double>(alpha_data)[alpha_data.sel->get_index(i)];
            state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
            state.iterations = duckdb::UnifiedVectorFormat::GetData<int>(iterations_data)[iterations_data.sel->get_index(i)];

            //state.Sigma += feature_value * feature_value;
            //state.C += feature_value * label_value;
            state.count++;
        }
    }
}

static void LinearRegressionCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (LinearRegressionState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<LinearRegressionState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->Sigma += state.Sigma;
        combined_ptr[i]->C += state.C;
        combined_ptr[i]->count += state.count;
    }
}

static void LinearRegressionFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];

        for (idx_t j = 0; j < state.iterations; j++) {
            auto gradient = getGradient(state.count, state.Sigma, state.C, state.theta, state.lambda);
            state.theta -= state.alpha * gradient;
        }

        duckdb::Value theta_value = duckdb::Value::CreateValue(state.theta);
        auto theta_pair = duckdb::Value::STRUCT({std::make_pair("key", "theta"), std::make_pair("value", theta_value)});
        duckdb::ListVector::PushBack(result, theta_pair);

        // Needed?
        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

duckdb::unique_ptr<duckdb::FunctionData> LinearRegressionBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto struct_type = duckdb::LogicalType::MAP(duckdb::LogicalType::VARCHAR, duckdb::LogicalType::DOUBLE);
    function.return_type = struct_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
} 

duckdb::AggregateFunction GetLinearRegressionFunction() {

    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE), // features
        duckdb::LogicalType::DOUBLE, // label
        duckdb::LogicalType::DOUBLE, // alpha
        duckdb::LogicalType::DOUBLE, // lambda
        duckdb::LogicalType::INTEGER // iterations
    };

    return duckdb::AggregateFunction(
        "linear_regression",                                                     // name     
        arg_types,                                                               // argument types
        duckdb::LogicalTypeId::MAP,                                              // return type
        duckdb::AggregateFunction::StateSize<LinearRegressionState>,             // state size
        duckdb::AggregateFunction::StateInitialize<LinearRegressionState, LinearRegressionFunction>, // state initialize
        LinearRegressionUpdate,                                                  // update
        LinearRegressionCombine,                                                 // combine
        LinearRegressionFinalize,                                                // finalize
        nullptr,                                                                 // simple update
        LinearRegressionBind,                                                    // bind
        duckdb::AggregateFunction::StateDestroy<LinearRegressionState, LinearRegressionFunction> // destroy
    );
}

void LinearRegression::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet linear_regression("linear_regression");
    linear_regression.AddFunction(GetLinearRegressionFunction());
    duckdb::CreateAggregateFunctionInfo info(linear_regression);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace quackml
