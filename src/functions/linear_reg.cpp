// UDAF to perform 1D regularised linear regression 
// Arguments to UDAF: one feature relation, one label relation, learning rate (alpha), regularisation parameter (lambda), number of iterations (iter)

#include "functions/linear_reg.hpp"

// Testing 
#include <iostream>
using std::cout;
using std::endl;

namespace quackml {

struct LinearRegFunction {
    // TODO: Are templates needed?
    template <class STATE>
    static void Initialize(STATE &state) {
        state.theta = 1.0;
        state.Sigma = 0.0;
        state.C = 0.0;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        return;
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

static void LinearRegUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegState **)sdata.data;

    auto &features = inputs[0];
    duckdb::UnifiedVectorFormat feature_data;
    features.ToUnifiedFormat(count, feature_data);

    auto &labels = inputs[1];
    duckdb::UnifiedVectorFormat label_data;
    labels.ToUnifiedFormat(count, label_data);

    for (idx_t i = 0; i < count; i++) {
        if (feature_data.validity.RowIsValid(feature_data.sel->get_index(i)) && 
            label_data.validity.RowIsValid(label_data.sel->get_index(i))) {
                auto &state = *states[sdata.sel->get_index(i)];
                // Get feature and labels
                auto feature = duckdb::UnifiedVectorFormat::GetData<double>(feature_data)[feature_data.sel->get_index(i)];
                auto label = duckdb::UnifiedVectorFormat::GetData<double>(label_data)[label_data.sel->get_index(i)];
                state.Sigma += feature * feature;
                state.C += feature * label;
        }
    }

}

static void LinearRegCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (LinearRegState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<LinearRegState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->Sigma += state.Sigma;
        combined_ptr[i]->C += state.C;
    }
}

static void LinearRegFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LinearRegState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];

        duckdb::Value Sigma = duckdb::Value::CreateValue(state.Sigma);
        duckdb::Value C = duckdb::Value::CreateValue(state.C);
        std::cout << "Sigma: " << Sigma.ToString() << "\n";
        std::cout << "C: " << C.ToString() << "\n";
    }
}

duckdb::unique_ptr<duckdb::FunctionData> LinearRegBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto return_type = duckdb::LogicalType::DOUBLE;
    function.return_type = return_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(return_type);
}

duckdb::AggregateFunction GetLinearRegFunction() {
    return duckdb::AggregateFunction(
        "linear_regression",                                                            // name
        {duckdb::LogicalType::DOUBLE},                                                  // argument types (unsure about this)
        duckdb::LogicalType::DOUBLE,                                                    // return type
        duckdb::AggregateFunction::StateSize<LinearRegState>,                           // state size
        duckdb::AggregateFunction::StateInitialize<LinearRegState, LinearRegFunction>,  // state initialize
        LinearRegUpdate,                                                                // update
        LinearRegCombine,                                                               // combine
        LinearRegFinalize,                                                              // finalize
        nullptr,                                                                        // simple update
        LinearRegBind,                                                                  // bind
        duckdb::AggregateFunction::StateDestroy<LinearRegState, LinearRegFunction>      // destroy
    );
}

void LinearRegression::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet linear_reg("linear_regression");
    linear_reg.AddFunction(GetLinearRegFunction());
    duckdb::CreateAggregateFunctionInfo info(linear_reg);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace quackml