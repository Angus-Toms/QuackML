#include "functions/linear_reg_ring.hpp"
#include "functions/linear_reg_utils.hpp"

#include <chrono>
#include <cmath>

namespace quackml {

struct LinearRegressionRingState {
public:
    double lambda; 
    LinRegRing *ring;
};

struct LinearRegressionRingFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.lambda = 0;
        state.ring = nullptr;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        if (state.ring) {
            delete state.ring;
        }
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
    auto states = (LinearRegressionRingState **)sdata.data;
    
    // Assume no outer GROUP BY clause is passed, therefore single state
    auto &state = *states[sdata.sel->get_index(0)];

    for (idx_t i = 0; i < count; i++) {
        // Iterate through distinct id values (if computing over a join)
        // count=1 if computing over cartesian product 
        auto ring_children = duckdb::ListValue::GetChildren(rings.GetValue(i));
        auto ring_elements = duckdb::ListValue::GetChildren(ring_children[0]);
        
        // d - first element in list 
        idx_t d = ring_elements[0].GetValue<int>();
        
        // count - second element in list
        idx_t observation_count = ring_elements[1].GetValue<int>();
        
        // sums - next d elements in list 
        std::vector<double> observation_sums;
        for (idx_t j = 2; j < d+2; j++) {
            observation_sums.push_back(ring_elements[j].GetValue<double>());
        }
        
        // covar - next d(d-1)/2 elements in list
        std::vector<std::vector<double>> observation_covar;
        idx_t covar_idx = d+2;
        for (idx_t j = 0; j < d; j++) {
            std::vector<double> row;
            for (idx_t k = j; k < d; k++) {
                row.push_back(ring_elements[covar_idx].GetValue<double>());
                covar_idx++;
            }
            observation_covar.push_back(row);
        }

        LinRegRing observation = LinRegRing(observation_count, observation_sums, observation_covar);

        for (idx_t j = 1; j < ring_children.size(); j++) {
            // Iterate through input relations 
            // Get count
            auto tuple_elements = duckdb::ListValue::GetChildren(ring_children[j]);

            // d - first element in list 
            idx_t d = tuple_elements[0].GetValue<int>();
            
            // count - second element in list
            idx_t tuple_count = tuple_elements[1].GetValue<int>();
            
            // sums - next d elements in list 
            std::vector<double> tuple_sums;
            for (idx_t j = 2; j < d+2; j++) {
                tuple_sums.push_back(tuple_elements[j].GetValue<double>());
            }
            
            // covar - next d(d-1)/2 elements in list
            std::vector<std::vector<double>> tuple_covar;
            idx_t covar_idx = d+2;
            for (idx_t j = 0; j < d; j++) {
                std::vector<double> row;
                for (idx_t k = j; k < d; k++) {
                    row.push_back(tuple_elements[covar_idx].GetValue<double>());
                    covar_idx++;
                }
                tuple_covar.push_back(row);
            }
            
            // Convert to LinRegRing 
            LinRegRing tuple_ring = LinRegRing(tuple_count, tuple_sums, tuple_covar);

            // Multiply into observation ring 
            auto observation_d = observation.sums.size();
            auto tuple_d = tuple_ring.sums.size();
            observation.padLower(tuple_d);
            tuple_ring.padUpper(observation_d);

            observation = observation * tuple_ring;
        }

        if (!state.ring) {
            // State not instantiated, allocate memory for state.ring and copy observation ring
            state.ring = new LinRegRing(observation);
            state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
        } else {
            // State exists, add observation into state
            *state.ring = *state.ring + observation;
        }
    }
}

static void LinearRegressionRingCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto state_ptr = (LinearRegressionRingState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<LinearRegressionRingState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *state_ptr[sdata.sel->get_index(i)];
        if (!state.ring) {
            // State not instantiated, skip
            continue;
        }
        if (!combined_ptr[i]->ring) {
            // Combined not instantiated, set combined to state
            combined_ptr[i]->ring = new LinRegRing(*state.ring);
            combined_ptr[i]->lambda = state.lambda;
        } else {
            // Both state and combined instantiated, add state to combined
            *combined_ptr[i]->ring = *combined_ptr[i]->ring * *state.ring;
        }
    }
}

static void LinearRegressionRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionRingState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    // Assume no outer GROUP BY clause is passed, therefore single state
    auto &state = *states[sdata.sel->get_index(0)];

    auto tuple_count = state.ring->count;
    auto sums = state.ring->sums;
    auto d = sums.size();
    auto covar = state.ring->covar;
    auto theta = new std::vector<std::vector<double>>(d-1, std::vector<double>(1, 1));
    auto lambda = state.lambda;

    // Slice Sigma, C from covar
    auto c = new std::vector<std::vector<double>>(d-1, std::vector<double>(1, 0));
    auto sigma = new std::vector<std::vector<double>>(d-1, std::vector<double>(d-1, 0));

    // Reconstruct the symmetric matrix
    for (int i = 1; i < d; i++) {
        (*c)[i-1][0] = covar[0][i];
        for (int j = i; j < d; j++) {
            (*sigma)[i-1][j-1] = covar[i][j-i];
            (*sigma)[j-1][i-1] = covar[i][j-i];
        }
    }

    // Gradient descent routine ------------------------------------------------
    // Convergence parameters
    auto max_iterations = 10000;
    idx_t iter = 0;
    double convergence_threshold = 1e-3;

    // Learning rate parameters
    double initial_learning_rate = 0.01 / std::sqrt(tuple_count);
    double decay_rate = 0.9;
    double decay_steps = 100;

    auto gradient = new std::vector<std::vector<double>>(d-1, std::vector<double>(1, 0));

    while (iter < max_iterations) {
        // Compute gradient
        for (idx_t i =0; i < d-1; i++) {
            // gradient = (1/n) * (sigma * theta - c) + (lambda * theta)
            (*gradient)[i][0] = 0;
            for (idx_t j = 0; j < d-1; j++) {
                (*gradient)[i][0] += (*sigma)[i][j] * (*theta)[j][0];
            }
            (*gradient)[i][0] -= (*c)[i][0]; // sigma * theta - c
            (*gradient)[i][0] += 1.0 / d-1; // (1/n) * (sigma * theta - c)
            (*gradient)[i][0] += lambda * (*theta)[i][0]; // (lambda * theta)
        }

        // Convergence testing and lr update 
        if (gradientNorm(*gradient) < convergence_threshold) {
            break;
        }
        double learning_rate = initial_learning_rate * std::pow(decay_rate, iter / decay_steps);

        // Update theta 
        for (idx_t i = 0; i < d-1; i++) {
            // theta -= (learning_rate * gradient)
            (*theta)[i][0] -= learning_rate * (*gradient)[i][0];
        }
        iter++;
    }

    delete c;
    delete sigma;
    delete gradient;
    
    // Create result vector
    for (idx_t j = 0; j < d-1; j++) {
        auto theta_value = duckdb::Value::CreateValue((*theta)[j][0]);
        duckdb::ListVector::PushBack(result, theta_value);
    }

    delete theta;

    // Set result list size 
    auto list_struct_data = duckdb::ListVector::GetData(result);
    list_struct_data[offset].length = duckdb::ListVector::GetListSize(result) - old_len;
    list_struct_data[offset].offset = old_len;
    old_len += list_struct_data[offset].length;

    result.Verify(tuple_count);
}

duckdb::unique_ptr<duckdb::FunctionData> LinearRegressionRingBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto list_type = duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
    function.return_type = list_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

duckdb::AggregateFunction GetLinearRegressionRingFunction() {
    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE)), // flat ringss
        duckdb::LogicalType::DOUBLE                                                        // lambda
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
