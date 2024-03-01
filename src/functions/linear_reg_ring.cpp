#include "functions/linear_reg_ring.hpp"
#include "functions/linear_reg_utils.hpp"

#include <chrono>
#include <cmath>

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
    // std::cout << "LinearRegressionRingUpdate called\n";
    
    // auto &rings = inputs[0];
    // auto &lambda = inputs[1];

    // duckdb::UnifiedVectorFormat rings_data;
    // duckdb::UnifiedVectorFormat lambda_data;
    // duckdb::UnifiedVectorFormat sdata;

    // rings.ToUnifiedFormat(count, rings_data);
    // lambda.ToUnifiedFormat(count, lambda_data);
    // state_vector.ToUnifiedFormat(count, sdata);

    // // MUNGO TODO: Support for GROUP BY clauses
    // auto states = (LinearRegressionRingState **)sdata.data;
    // auto &state = *states[sdata.sel->get_index(0)];

    // auto ring_children = duckdb::ListValue::GetChildren(rings.GetValue(0));
    // auto ring_count = ring_children.size();

    // for (idx_t i = 0; i < ring_count; i++) {
    //     auto ring = new LinearRegressionRingElement(ring_children[i]);
    //     if (!state.ring) {
    //         // Initialize ring and model hyperparameters if not already 
    //         state.ring = ring;
    //         state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
    //     } else {
    //         // Add padding to state 
    //         auto state_d = state.ring->get_d();
    //         auto ring_d = ring->get_d();
    //         state.ring->pad_lower(ring_d);
    //         ring->pad_upper(state_d);

    //         // Update state 
    //         auto new_ring = (*state.ring) * (*ring);
    //         state.ring = &new_ring; // Could this be issue?
    //         state.ring->Print();

    //         std::cout << "Combined ring address:" << state.ring << "\n";
    //         std::cout << "After multiplication, covar address: " << state.ring->get_covar() << "\n";
    //         for (auto &row : duckdb::ListValue::GetChildren(*(state.ring->get_covar()))) {
    //             for (auto &col : duckdb::ListValue::GetChildren(row)) {
    //                 std::cout << col.GetValue<double>() << " ";
    //             }
    //             std::cout << "\n";
    //         }
    //     }
    // }
    std::cout << "LinearRegressionRingUpdate called\n";

    auto &ring_input = inputs[0];
    auto rings = duckdb::ListValue::GetChildren(ring_input.GetValue(0));
    auto ring_count = rings.size();

    auto &lambda = inputs[1];
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat sdata;
    lambda.ToUnifiedFormat(count, lambda_data);
    state_vector.ToUnifiedFormat(count, sdata);

    auto states = (LinearRegressionRingState **)sdata.data;
    auto &state = *states[sdata.sel->get_index(0)]; 

    for (idx_t i = 0; i < ring_count; i++) {
        auto ring = new LinearRegressionRingElement(rings[i]);
        if (!state.ring) {
            state.ring = ring;
        } else {
            // Padding rings, state ring must always be left operand 
            auto state_d = state.ring->get_d();
            auto ring_d = ring->get_d();
            state.ring->pad_lower(ring_d);
            ring->pad_upper(state_d);

            // Multiply rings, use copy constructor to dynamically allocate object 
            auto combined = new LinearRegressionRingElement((*state.ring) * (*ring));
            combined->Print();
            // Taking address of temp object?
            state.ring = combined;
        }
    }   
}

static void LinearRegressionRingCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    std::cout << "LinearRegressionRingCombine called\n";
}

static void LinearRegressionRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    // std::cout << "LinearRegressionRingFinalize called\n";

    // duckdb::UnifiedVectorFormat sdata;
    // state_vector.ToUnifiedFormat(count, sdata);
    // auto states = (LinearRegressionRingState **)sdata.data;

    // auto &mask = duckdb::FlatVector::Validity(result);
    // auto old_len = duckdb::ListVector::GetListSize(result);

    // for (idx_t i = 0; i < count; i++) {
    //     const auto rid = i + offset;
    //     auto d = 0;
    //     auto &state = *states[sdata.sel->get_index(i)];
    //     auto ring = state.ring;
    //     std::cout << "Finalize ring address: " << ring << "\n";

    //     if (!state.theta) {
    //         d = ring->get_d();
    //         std::cout << "Ring d: " << d << "\n";
    //         state.theta = new std::vector<std::vector<double>>(d, std::vector<double>(1, 0.0));
    //     }

    //     // Slice C, sigma from covariance matrix
    //     auto covariance = ring->get_covar();
    //     std::cout << "Covar address: " << covariance << "\n";
    //     std::cout << "Printing covar:\n";
    //     for (auto row : duckdb::ListValue::GetChildren(*covariance)) {
    //         for (auto col : duckdb::ListValue::GetChildren(row)) {
    //             std::cout << col.GetValue<double>() << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "Covar printed\n";

    //     auto covariance_children = duckdb::ListValue::GetChildren(*covariance);
        
    //     std::vector<std::vector<double>> c;
    //     std::vector<std::vector<double>> sigma;

    //     // First row of covariance matrix contains c
    //     std::cout << "Extracting C, sigma\n";
    //     auto covar_row = duckdb::ListValue::GetChildren(covariance_children[0]);
    //     for (idx_t i = 1; i < d; i++) {
    //         c.push_back({covar_row[i].GetValue<double>()});
    //     }
    //     std::cout << "C extracted\n";
    //     printMatrix(c);

    //     // Extract sigma
    //     for (idx_t i = 1; i < d; i++) {
    //         auto sigma_row = duckdb::ListValue::GetChildren(covariance_children[i]);
    //         std::vector<double> sigma_row_vec;
    //         sigma_row.erase(sigma_row.begin());
    //         for (auto &sigma_val : sigma_row) {
    //             sigma_row_vec.push_back(sigma_val.GetValue<double>());
    //         }
    //     }
    //     std::cout << "Sigma extracted\n";
    //     printMatrix(sigma);

    //     std::cout << "C:\n";
    //     for (auto &row : c) {
    //         for (auto &col : row) {
    //             std::cout << col << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "Sigma:\n";
    //     for (auto &row : sigma) {
    //         for (auto &col : row) {
    //             std::cout << col << " ";
    //         }
    //         std::cout << "\n";
    //     }

    //     // Convergence parameters
    //     auto max_iterations = 10000;
    //     idx_t iter = 0;
    //     double convergence_threshold = 1e-5;

    //     // Learning rate parameters
    //     double initial_learning_rate = 0.1 / std::sqrt(state.count);
    //     double decay_rate = 0.9; 
    //     double decay_steps = 100; 

    //     while (iter < max_iterations) {
    //         auto gradient = getGradientND(sigma, c, *state.theta, state.lambda);
    //         if (gradientNorm(gradient) < convergence_threshold) { break; }
    //         double learning_rate = initial_learning_rate * pow(decay_rate, iter / decay_steps);
    //         // learning_rate * gradient 
    //         matrixScalarMultiply(gradient, learning_rate, gradient);
    //         // theta -= (learning_rate * gradient)
    //         matrixSubtract(*state.theta, gradient, *state.theta);
    //         iter++;
    //     }

    //     std::cout << "GD performed in " << iter << " iterations\n";

    //     // Create weight result list 
    //     for (idx_t j = 0; j < state.theta->size(); j++) {
    //         auto theta_value = duckdb::Value::CreateValue((*state.theta)[j][0]);
    //         duckdb::ListVector::PushBack(result, theta_value);
    //     }
    //     // Set size of result
    //     auto list_struct_data = duckdb::ListVector::GetData(result);
    //     list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
    //     list_struct_data[rid].offset = old_len;
    //     old_len += list_struct_data[rid].length;
    // }
    // result.Verify(count);
    std::cout << "LinearRegressionRingFinalize called\n";
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