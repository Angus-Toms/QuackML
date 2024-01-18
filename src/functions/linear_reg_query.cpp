// Ridge linear regression with L2 cost function, optimised with GD using covariance matrix.
// Covariance matrix calculations done through batch of aggregate queries 

#include "functions/linear_reg_query.hpp"
#include <iostream>

namespace quackml {

float getGradient1D(idx_t n, double Sigma, double C, double theta, double lambda) {
    // Calculate cost function gradient using covariance matrix
    return (1.0 / n) * ((Sigma * theta - C) + (lambda * theta));
};

struct LinearRegressionQueryState {
    idx_t count;

    double theta;
    double sigma;
    double c;

    double alpha;
    double lambda;
    idx_t iter;
};

struct LinearRegressionQueryFunction {
    template <class STATE> 
    static void Initialize(STATE &state) {
        state.count = 0;

        state.theta = 1.0;
        state.sigma = 0.0;
        state.c = 0.0;

        state.alpha = 0.0;
        state.lambda = 0.0;
        state.iter = 0;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        return;
    }

    static bool IgnoreNull() {
        return true;
    }
};

static void LinearRegressionQueryUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &features = inputs[0];
    auto &labels = inputs[1];
    auto &alpha = inputs[2];
    auto &lambda = inputs[3];
    auto &iter = inputs[4];

    duckdb::UnifiedVectorFormat feature_data;
    duckdb::UnifiedVectorFormat label_data;
    duckdb::UnifiedVectorFormat alpha_data;
    duckdb::UnifiedVectorFormat lambda_data;
    duckdb::UnifiedVectorFormat iter_data;

    features.ToUnifiedFormat(count, feature_data);
    labels.ToUnifiedFormat(count, label_data);
    alpha.ToUnifiedFormat(count, alpha_data);
    lambda.ToUnifiedFormat(count, lambda_data);
    iter.ToUnifiedFormat(count, iter_data);

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionQueryState **)sdata.data;
    
    // TODO: Update for multiple states in future
    auto state = states[sdata.sel->get_index(0)];
    state->alpha = duckdb::UnifiedVectorFormat::GetData<double>(alpha_data)[alpha_data.sel->get_index(0)];
    state->lambda = duckdb::UnifiedVectorFormat::GetData<double>(lambda_data)[lambda_data.sel->get_index(0)];
    state->iter = duckdb::UnifiedVectorFormat::GetData<int64_t>(iter_data)[iter_data.sel->get_index(0)];

    // Instantiate DB 
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    con.Query("CREATE TABLE t (feature DOUBLE, label DOUBLE);");

    // Insert data
    std::string values = "INSERT INTO t VALUES";
    for (idx_t i = 0; i < count; i++) {
        if (feature_data.validity.RowIsValid(feature_data.sel->get_index(i)) && label_data.validity.RowIsValid(label_data.sel->get_index(i))) {
            auto feature = duckdb::UnifiedVectorFormat::GetData<double>(feature_data)[feature_data.sel->get_index(i)];
            auto label = duckdb::UnifiedVectorFormat::GetData<double>(label_data)[label_data.sel->get_index(i)];
            values += "(" + std::to_string(feature) + ", " + std::to_string(label) + ")";
            if (i != count-1) {
                values += ", ";
            }
        }
    }
    values += ";";
    con.Query(values);

    // TODO: Support for GROUP BY, currently calling to single state
    // Get sigma by query
    state->sigma = con.Query("SELECT SUM(feature * feature) FROM t;")->GetValue<double>(0, 0);

    // Get c by query
    state->c = con.Query("SELECT SUM(feature * label) FROM t;")->GetValue<double>(0, 0);

    // Increase count
    state->count += count;

}

static void LinearRegressionQueryCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (LinearRegressionQueryState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<LinearRegressionQueryState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->count += state.count;
        combined_ptr[i]->sigma += state.sigma;
        combined_ptr[i]->c += state.c;

        combined_ptr[i]->alpha = state.alpha;
        combined_ptr[i]->lambda = state.lambda;
        combined_ptr[i]->iter = state.iter;
    }
}

static void LinearRegressionQueryFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegressionQueryState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];
        
        // Gradient descent 
        for (idx_t j = 0; j < state.iter; j++) {
            auto gradient = getGradient1D(state.count, state.sigma, state.c, state.theta, state.lambda);
            state.theta -= state.alpha * gradient;
        }

        auto theta_pair = duckdb::Value::STRUCT({std::make_pair("key", "theta"), std::make_pair("value", duckdb::Value::CreateValue(state.theta))});
        duckdb::ListVector::PushBack(result, theta_pair);

        // Needed?
        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

duckdb::unique_ptr<duckdb::FunctionData> LinearRegressionQueryBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto struct_type = duckdb::LogicalType::MAP(duckdb::LogicalType::VARCHAR, duckdb::LogicalType::DOUBLE);
    function.return_type = struct_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

duckdb::AggregateFunction GetLinearRegressionQueryFunction() {
    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::DOUBLE, //features 
        duckdb::LogicalType::DOUBLE, //labels
        duckdb::LogicalType::DOUBLE, //alpha
        duckdb::LogicalType::DOUBLE, //lambda
        duckdb::LogicalType::BIGINT //iter
    };

    return duckdb::AggregateFunction(
        "linear_regression_query",                                                  // name
        arg_types,                                                                  // argument types
        duckdb::LogicalTypeId::MAP,                                                 // return type
        duckdb::AggregateFunction::StateSize<LinearRegressionQueryState>,           // state size
        duckdb::AggregateFunction::StateInitialize<LinearRegressionQueryState, LinearRegressionQueryFunction>,  // state initialize
        LinearRegressionQueryUpdate,                                                // update
        LinearRegressionQueryCombine,                                               // combine
        LinearRegressionQueryFinalize,                                              // finalize
        nullptr,                                                                    // simple update
        LinearRegressionQueryBind,                                                  // bind
        duckdb::AggregateFunction::StateDestroy<LinearRegressionQueryState, LinearRegressionQueryFunction>      // state destroy     
    );
}

void LinearRegressionQuery::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet linear_regression_query("linear_regression_query");
    linear_regression_query.AddFunction(GetLinearRegressionQueryFunction());
    duckdb::CreateAggregateFunctionInfo linear_regression_query_info(linear_regression_query);
    catalog.CreateFunction(*conn.context, linear_regression_query_info);
}

} // namespace quackml