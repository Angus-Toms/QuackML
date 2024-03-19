#include "functions/linear_reg_ring.hpp"
#include "functions/linear_reg_utils.hpp"

#include <chrono>
#include <cmath>

// MUNGO TODO: Add bias term and issues with state dimensions if we're removing 
// one row of labels

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

// TODO: Count?

static void LinearRegressionRingUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    std::cout << "LinearRegressionRingUpdate called\n";
    auto &rings = inputs[0];
    auto &lambda = inputs[1];
    duckdb::UnifiedVectorFormat rings_data;
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat sdata;

    rings.ToUnifiedFormat(count, rings_data);
    lambda.ToUnifiedFormat(count, lambda_data);
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionRingState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        //std::cout << "Processing row " << i << "\n";
        // Multiply all rings together, 
        // If computing across a join, n lists of rings will be passed 
        // and their resulting products will be summed (for n distinct values
        // of the join attribute)
        auto ring_children = duckdb::ListValue::GetChildren(rings.GetValue(i));
        auto observation_ring = new LinearRegressionRingElement(ring_children[0]);
        for (idx_t j = 1; j < ring_children.size(); j++) {
            //std::cout << "Processing ring " << j << "\n";
            //printMatrix(ring_children[j])
            auto ring = new LinearRegressionRingElement(ring_children[j]);
            auto ring_d = ring->get_d();
            auto observation_d = observation_ring->get_d();
            observation_ring->pad_lower(ring_d);
            ring->pad_upper(observation_d);
            observation_ring = new LinearRegressionRingElement((*observation_ring) * (*ring));
        }

        auto &state = *states[sdata.sel->get_index(i)];
        if (!state.ring) {
            state.ring = observation_ring;
            state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
        } else {
            state.ring = new LinearRegressionRingElement((*state.ring) + (*observation_ring));
        }
    }
}

static void NewLinearRegressionRingUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &rings = inputs[0];
    auto &lambda = inputs[1];
    duckdb::UnifiedVectorFormat rings_data;
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat sdata;

    rings.ToUnifiedFormat(count, rings_data);
    lambda.ToUnifiedFormat(count, lambda_data);
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionRingState **)sdata.data;

    // Find all children for a given state
}

static void LinearRegressionRingCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto state_ptr = (LinearRegressionRingState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<LinearRegressionRingState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *state_ptr[sdata.sel->get_index(i)];
        if (!state.ring) {
            continue; 
        }

        if (!combined_ptr[i]->ring) {
            // No combined ring exists yet, just set it to the state ring (guaranteed to exist)
            combined_ptr[i]->ring = state.ring;
        } else {
            // Combined ring does exist, sum state ring into it
            combined_ptr[i]->ring = new LinearRegressionRingElement((*combined_ptr[i]->ring) + (*state.ring));
            combined_ptr[i]->lambda = state.lambda;
        }
    }
}

static void LinearRegressionRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionRingState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        auto rid = i + offset;

        auto &state = *states[sdata.sel->get_index(i)];
        auto ring = state.ring;
        auto d = ring->get_d();

        // Instantiate theta if not already 
        if (!state.theta) {
            state.theta = new std::vector<std::vector<double>>(d-1, std::vector<double>(1, 0.0));
        }
        
        // Extract Sigma, C 
        std::vector<std::vector<double>> sigma;
        std::vector<std::vector<double>> c;

        auto covar = ring->get_covar();
        auto covar_children = duckdb::ListValue::GetChildren(*covar);
        auto covar_first_row = duckdb::ListValue::GetChildren(covar_children[0]);

        for (idx_t i = 1; i < d; i++) {
            c.push_back({covar_first_row[i].GetValue<double>()});
            auto row_children = duckdb::ListValue::GetChildren(covar_children[i]);
            std::vector<double> row_vec;
            for (idx_t j = 1; j < d; j++) {
                row_vec.push_back(row_children[j].GetValue<double>());
            }
            sigma.push_back(row_vec);
        }

        // Gradient descent
        // Convergence parameters
        auto max_iterations = 10000;
        idx_t iter = 0;
        double convergence_threshold = 1e-3;

        // Learning rate parameters
        auto count = state.ring->get_count()->GetValue<double>();
        double initial_learning_rate = count < 100 ? 0.05 : 5e-9 / std::sqrt(count);
        double decay_rate = 0.95; 
        double decay_steps = 100; 


        while (iter < max_iterations) {
            auto gradient = getGradientND(sigma, c, *state.theta, state.lambda);
            if (gradientNorm(gradient) < convergence_threshold) { break; }
            double learning_rate = initial_learning_rate * pow(decay_rate, iter / decay_steps);
            // learning_rate * gradient
            matrixScalarMultiply(gradient, learning_rate, gradient);
            // theta -= (learning_rate * gradient)
            matrixSubtract(*state.theta, gradient, *state.theta);
            iter++;
        }
        std::cout << "GD performed in " << iter << " iterations\n";

        // Create result vector
        for (idx_t j = 0; j < d-1; j++) {
            auto theta_value = duckdb::Value::CreateValue((*state.theta)[j][0]);
            duckdb::ListVector::PushBack(result, theta_value);
        }

        // Set result list size 
        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

duckdb::unique_ptr<duckdb::FunctionData> LinearRegressionRingBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto list_type = duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
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