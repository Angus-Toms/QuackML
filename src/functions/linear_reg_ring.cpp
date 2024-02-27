#include "functions/linear_reg_ring.hpp"

// Testing:
#include <chrono>
#include <iostream>

namespace quackml {

struct LinearRegressionRingState {
    idx_t count;
    double lambda;

    LinearRegressionRingElement *ring;
    std::vector<std::vector<double>> *theta;
};

struct LinearRegressionRingFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.count = 0;
        state.lambda = 0;

        state.ring = nullptr;
        state.theta = nullptr;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        // MUNGO TODO: Implement
    }

    static bool IgnoreNull() {
        return true;
    }
};

static void LinearRegressionRingUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &rings = inputs[0];
    auto &lambda = inputs[1];

    duckdb::UnifiedVectorFormat rings_data;
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat sdata;

    rings.ToUnifiedFormat(count, rings_data);
    lambda.ToUnifiedFormat(count, lambda_data);
    state_vector.ToUnifiedFormat(count, sdata);

    // MUNGO TODO: Support for GROUP BY clauses
    auto states = (LinearRegressionRingState **)sdata.data;
    auto state = *states[sdata.sel->get_index(0)];

    auto ring_children = duckdb::ListValue::GetChildren(rings.GetValue(0));
    auto ring_count = ring_children.size();

    for (idx_t i = 0; i < ring_count; i++) {
        auto ring = new LinearRegressionRingElement(ring_children[i]);

        if (!state.ring) {
            // Initialize ring and model hyperparameters if not already 
            state.ring = ring;
            state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
        } else {
            // Add padding to state 
            auto state_d = state.ring->get_d();
            auto ring_d = ring->get_d();

            state.ring->pad_lower(ring_d);
            ring->pad_upper(state_d);

            // Update state 
            auto new_ring = (*state.ring) * (*ring);
            state.ring = &new_ring;

            state.ring->Print();
        }
    }
}

static void LinearRegressionRingCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    std::cout << "LinearRegressionRingCombine called\n";
}

static void LinearRegressionRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    std::cout << "LinearRegressionRingFinalize called\n";

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionRingState **)sdata.data;
    auto state = *states[sdata.sel->get_index(0)];
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);
                                                                                
    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;

        if (!state.theta) {
            state.theta = new std::vector<std::vector<double>>(state.ring->get_d(), std::vector<double>(1, 0.0));
            std::cout << "theta initialized\n";
        }

        // Gradient descent 
        for (idx_t j = 0; j < 10000; j++) {
            // TODO: Convert state.ring to std::vector 
            // TODO: Create c 
            // auto gradient = getGradientND(*state.ring, *state.c, *state.theta, state.lambda);
            // matrixScalarMultiply(gradient, state.alpha, gradient);
            // matrixSubtract(*state.theta, gradient, *state.theta);
            auto hello = 0;
            std::cout << "GD iteration " << j << "\n";
        }

        std::cout << "GD done\n";

        // Create weight result list 
        for (idx_t j = 0; j < state.theta->size(); j++) {
            auto theta_value = duckdb::Value::CreateValue((*state.theta)[j][0]);
            duckdb::ListVector::PushBack(result, theta_value);
        }

        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

duckdb::unique_ptr<duckdb::FunctionData> LinearRegressionRingBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto list_type = duckdb::LogicalType::LIST(LinearRegressionRingType);
    function.return_type = list_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

duckdb::AggregateFunction GetLinearRegressionRingFunction() {
    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(LinearRegressionRingType),    // list of rings
        duckdb::LogicalType::DOUBLE                             // lambda
    };

    return duckdb::AggregateFunction(
        "linear_regression_ring",                                                                               // name
        arg_types,                                                                                              // argument types 
        duckdb::LogicalTypeId::LIST,                                                                            // return type
        duckdb::AggregateFunction::StateSize<LinearRegressionRingState>,                                        // state size
        duckdb::AggregateFunction::StateInitialize<LinearRegressionRingState, LinearRegressionRingFunction>,    // state initialize
        LinearRegressionRingUpdate,                                                                             // update
        LinearRegressionRingCombine,                                                                            // combine
        LinearRegressionRingFinalize,                                                                           // finalize                    
        nullptr,                                                                                                // simple update          
        LinearRegressionRingBind,                                                                               // bind 
        duckdb::AggregateFunction::StateDestroy<LinearRegressionRingState, LinearRegressionRingFunction>        // destructor
    );
}

void LinearRegressionRing::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet linear_regression_ring("linear_regression_ring");
    linear_regression_ring.AddFunction(GetLinearRegressionRingFunction());
    duckdb::CreateAggregateFunctionInfo info(linear_regression_ring);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace quackml