// UDAF to perform regularised linear regression 
// MUNGO TODO: Debug issue with seg faults from multiple calls to same table - destroy function not working?
// Think issue is calling multiple linear regressions with different hyperparameters?
#include "functions/linear_reg.hpp"
#include "functions/linear_reg_utils.hpp"
#include <chrono>

namespace quackml {

struct LinearRegressionState {
    idx_t count;
    idx_t d;

    std::vector<std::vector<double>> *sigma;
    std::vector<std::vector<double>> *c;
    std::vector<std::vector<double>> *theta;

    double lambda;
};

struct LinearRegressionFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.count = 0;

        state.sigma = nullptr;
        state.c = nullptr;
        state.theta = nullptr;

        state.lambda = 0.0;
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
    auto &lambda = inputs[2];
    duckdb::UnifiedVectorFormat feature_data;
    duckdb::UnifiedVectorFormat label_data;
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat sdata;

    feature.ToUnifiedFormat(count, feature_data);
    label.ToUnifiedFormat(count, label_data);
    lambda.ToUnifiedFormat(count, lambda_data);
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        //std::cout << "Processing row: " << i << "\n";
        if (feature_data.validity.RowIsValid(feature_data.sel->get_index(i)) && label_data.validity.RowIsValid(label_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            auto feature_vector = duckdb::ListValue::GetChildren(feature.GetValue(i));
            feature_vector.push_back(duckdb::Value::DOUBLE(1)); // Add bias term
            state.d = feature_vector.size();
            state.lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(i)];
            auto label_value = duckdb::UnifiedVectorFormat::GetData<double>(label_data)[label_data.sel->get_index(i)];

            // Initialise sigma, c if empty
            if (!state.sigma) {
                // If sigma is empty, c will be also so only one check needed
                state.sigma = new std::vector<std::vector<double>>();
                state.c = new std::vector<std::vector<double>>();
                // Add extra row/column to account for bias term
                for (idx_t j = 0; j < state.d; j++) {
                    state.sigma->push_back(std::vector<double>(state.d, 0));
                    state.c->push_back(std::vector<double>(1, 0));
                }
            }

            // Update sigma, c
            for (idx_t j = 0; j < state.d; j++) {
                auto feature_j = feature_vector[j].GetValue<double>();
                (*state.c)[j][0] += feature_j * label_value;
                for (idx_t k = j; k < state.d; k++) {
                    (*state.sigma)[j][k] += feature_j * feature_vector[k].GetValue<double>();
                    (*state.sigma)[k][j] = (*state.sigma)[j][k];
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
        if (!state.sigma) {
            continue;
        }

        if (!combined_ptr[i]->sigma) {
            combined_ptr[i]->sigma = new std::vector<std::vector<double>>(state.d, std::vector<double>(state.d, 0));
            combined_ptr[i]->c = new std::vector<std::vector<double>>(state.d, std::vector<double>(1, 0));
        }

        matrixAdd(*state.sigma, *combined_ptr[i]->sigma, *combined_ptr[i]->sigma);
        matrixAdd(*state.c, *combined_ptr[i]->c, *combined_ptr[i]->c);

        combined_ptr[i]->count += state.count;
        combined_ptr[i]->d = state.d;
        combined_ptr[i]->lambda = state.lambda;
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
            state.theta = new std::vector<std::vector<double>>(state.d, std::vector<double>(1, 0.0));
        }

        // MUNGO TODO: Rewrite old GD routine
        // Test GD routine and see how much time learning rate decay and convergence testing saves
        // Just make new finalize function, and then new agg function using it
        // Also linear algebra libraries? (if time)


        // Convergence parameters
        auto max_iterations = 10000;
        idx_t iter = 0;
        double convergence_threshold = 1e-5;

        // Learning rate parameters
        double initial_learning_rate = 0.1 / std::sqrt(state.count);
        double decay_rate = 0.9; 
        double decay_steps = 100; 

        while (iter < max_iterations) {
            auto gradient = getGradientND(*state.sigma, *state.c, *state.theta, state.lambda);
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
        for (idx_t j = 0; j < state.d; j++) {
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
        duckdb::LogicalType::DOUBLE                             // lambda
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
