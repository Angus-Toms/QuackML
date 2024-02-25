// UDAF to perform regularised linear regression 
// MUNGO TODO: Debug group by clauses 
// MUNGO TODO: Debug issue with seg faults from multiple calls to same table - destroy function not working?
// Think issue is calling multiple linear regressions with different hyperparameters?
// MUNGO TODO: Add bias term

#include "functions/linear_reg.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using std::vector;
using std::to_string;

namespace quackml {

void matrixScalarMultiply(std::vector<std::vector<double>> &matrix, float scalar, std::vector<std::vector<double>> &result) {
    // Multiply each element of a matrix by a scalar
    for (size_t i=0; i<matrix.size(); i++) {
        for (size_t j=0; j<matrix[0].size(); j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    };
};

void matrixMultiply(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result) {
    // Multiply two matrices
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix2[0].size(); j++) {
            for (size_t k=0; k<matrix2.size(); k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
};

void matrixSubtract(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result) {
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix1[0].size(); j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
};

void matrixAdd(std::vector<std::vector<double>> &matrix1, std::vector<std::vector<double>> &matrix2, std::vector<std::vector<double>> &result) {
    for (size_t i=0; i<matrix1.size(); i++) {
        for (size_t j=0; j<matrix1[0].size(); j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
};

std::vector<std::vector<double>> getGradientND(std::vector<std::vector<double>> &sigma, std::vector<std::vector<double>> &c, std::vector<std::vector<double>> &theta, double lambda) {    
    size_t n = sigma.size();
    size_t d = sigma[0].size();

    // sigma * theta
    std::vector<std::vector<double>> result(d, std::vector<double>(1, 0));
    matrixMultiply(sigma, theta, result);

    // sigma * theta - c
    matrixSubtract(result, c, result);

    // (1/n) * (sigma * theta - c)
    matrixScalarMultiply(result, (1.0 / n), result);

    // lambda * theta
    std::vector<std::vector<double>> regularizer(d, std::vector<double>(1, 0));
    matrixScalarMultiply(theta, lambda, regularizer);

    // (1/n) * (sigma * theta - c) + (lambda * theta)
    matrixAdd(result, regularizer, result);

    return result;
};


struct LinearRegressionState {
    idx_t count;
    idx_t d;

    std::vector<std::vector<double>> *sigma;
    std::vector<std::vector<double>> *c;
    std::vector<std::vector<double>> *theta;

    double alpha;
    double lambda;
    idx_t iterations;
};

struct LinearRegressionFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.count = 0;

        state.sigma = nullptr;
        state.c = nullptr;
        state.theta = nullptr;

        state.alpha = 0.01;
        state.lambda = 0.0;
        state.iterations = 1000;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
    }

    static bool IgnoreNull() {
        return true;
    }
};

static void LinearRegressionUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    std::cout << "Update called\n";

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
        std::cout << "Processing row: " << i << "\n";
        if (feature_data.validity.RowIsValid(feature_data.sel->get_index(i)) && label_data.validity.RowIsValid(label_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            auto feature_vector = duckdb::ListValue::GetChildren(feature.GetValue(i));
            state.d = feature_vector.size();

            state.alpha = duckdb::UnifiedVectorFormat::GetData<double>(alpha_data)[alpha_data.sel->get_index(i)];
            state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
            state.iterations = duckdb::UnifiedVectorFormat::GetData<int>(iterations_data)[iterations_data.sel->get_index(i)];
            auto label_value = duckdb::UnifiedVectorFormat::GetData<double>(label_data)[label_data.sel->get_index(i)];

            // Initialise sigma, c if empty
            if (!state.sigma) {
                std::cout << "Instantiating sigma for state: " << &state << "\n";
                // If sigma is empty, c will be also so only one check needed
                state.sigma = new std::vector<std::vector<double>>();
                state.c = new std::vector<std::vector<double>>();
                for (idx_t j = 0; j < state.d; j++) {
                    state.sigma->push_back(std::vector<double>(state.d, 0));
                    state.c->push_back(std::vector<double>(1, 0));
                }
            } else {
                std::cout << "Sigma already exists for state " << &state << "\n";
            }

            // Update sigma, c
            std::cout << "Updating sigma and c\n";
            for (idx_t j = 0; j < state.d; j++) {
                auto feature_j = feature_vector[j].GetValue<double>();
                (*state.c)[j][0] += feature_j * label_value;
                for (idx_t k = j; k < state.d; k++) {
                    auto feature_product = feature_j * feature_vector[k].GetValue<double>();
                    (*state.sigma)[j][k] = feature_product;
                    (*state.sigma)[k][j] = feature_product;
                }
            }
            state.count++;
        }
    }
}

static void LinearRegressionCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    std::cout << "Combined called\n";

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (LinearRegressionState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<LinearRegressionState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        if (!combined_ptr[i]->sigma) {
            //std::cout << "Instantiating sigma for state: " << &state << "\n";
            combined_ptr[i]->sigma = new std::vector<std::vector<double>>(state.d, std::vector<double>(state.d, 0));
            combined_ptr[i]->c = new std::vector<std::vector<double>>(state.d, std::vector<double>(1, 0));
        }
        
        combined_ptr[i]->alpha = state.alpha;
        combined_ptr[i]->lambda = state.lambda;
        combined_ptr[i]->iterations = state.iterations;
    }
}

static void LinearRegressionFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    std::cout << "Finalize called\n";

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];
        
        // Instantiate theta if not already
        if (!state.theta) {
            std::cout << "Instantiating theta for state: " << &state << "\n";
            state.theta = new std::vector<std::vector<double>>(state.d, std::vector<double>(1, 0.0));
        } else {
            std::cout << "Theta already exists for state " << &state << "\n";
        }

        // Gradient descent
        for (idx_t j = 0; j < state.iterations; j++) {
            auto gradient = getGradientND(*state.sigma, *state.c, *state.theta, state.lambda);
            // learning_rate * gradient 
            matrixScalarMultiply(gradient, state.alpha, gradient);
            // theta -= (learning_rate * gradient)
            matrixSubtract(*state.theta, gradient, *state.theta);
        }

        std::cout << "GD performed\n";

        // Create weight result pairs
        // for (idx_t j = 0; j < state.d; j++) {
        //     auto theta_value = duckdb::Value::CreateValue((*state.theta)[j][0]);
        //     auto key = "theta_" + std::to_string(j);
        //     auto theta_pair = duckdb::Value::STRUCT({std::make_pair("key", key), std::make_pair("value", theta_value)});
        //     duckdb::ListVector::PushBack(result, theta_pair);
        // }

        // Create result vector 
        for (idx_t j = 0; j < state.d; j++) {
            std::cout << "Pushing back theta value: " << (*state.theta)[j][0] << "\n";
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

duckdb::unique_ptr<duckdb::FunctionData> LinearRegressionBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto list_type = duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
    function.return_type = list_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
} 

duckdb::AggregateFunction GetLinearRegressionFunction() {
    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE), // features
        duckdb::LogicalType::DOUBLE,                            // label
        duckdb::LogicalType::DOUBLE,                            // alpha
        duckdb::LogicalType::DOUBLE,                            // lambda
        duckdb::LogicalType::INTEGER                            // iterations
    };

    return duckdb::AggregateFunction(
        "linear_regression",                                                                        // name     
        arg_types,                                                                                  // argument types
        duckdb::LogicalTypeId::LIST,                                                                // return type
        duckdb::AggregateFunction::StateSize<LinearRegressionState>,                                // state size
        duckdb::AggregateFunction::StateInitialize<LinearRegressionState, LinearRegressionFunction>,// state initialize
        LinearRegressionUpdate,                                                                     // update
        LinearRegressionCombine,                                                                    // combine
        LinearRegressionFinalize,                                                                   // finalize
        nullptr,                                                                                    // simple update
        LinearRegressionBind,                                                                       // bind
        duckdb::AggregateFunction::StateDestroy<LinearRegressionState, LinearRegressionFunction>    // destroy
    );
}

void LinearRegression::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet linear_regression("linear_regression");
    linear_regression.AddFunction(GetLinearRegressionFunction());
    duckdb::CreateAggregateFunctionInfo info(linear_regression);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace quackml
