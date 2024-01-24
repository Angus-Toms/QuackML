// This aggregate function converts an arbitrary number of fields in a relation
// to a single linear regression ring element. These elements can then be 
// combined by the linear regression compute function.

#include <iostream>
#include "duckdb.hpp"
#include "../types/linear_reg_ring.cpp"

namespace quackml {

struct ToRingState {
    LinearRegressionRingElement *ringElement;
};

struct ToRingFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.element = nullptr;
    }

    template <class STATE>
    static void Destroy(STATE &state) {
        if (state.element) {
            delete state.element;
        }
    }

    static bool IgnoreNull() {
        return true;
    }
};

static void ToRingUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &features = inputs[0];
    duckdb::UnifiedVectorFormat features_data;
    features.ToUnifiedFormat(count, features_data);

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ToRingState **)sdata.data;

    state.ringElemtent = new LinearRegressionRingElement(features.size());

    for (idx_t i = 0; i < count; i++) {
        if (feature_data.validity.RowIsValid(feature_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            auto feature_vector = duckdb::ListValue::GetChildren(feature.GetValue(i));
            
            auto tuple_ring = new LinearRegressionRingElement(feature_vector);
            state.ringElement = state.ringElement + tuple_ring;

        }
    }
}

static void ToRingCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (ToRingState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<ToRingState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->ringElement = combined_ptr[i]->ringElement + state.ringElement;
    }
}

static void ToRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ToRingState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        result.SetValue(i, state.ringElement);
    }
}

duckdb::unique_ptr<duckdb::FunctionData> ToRingBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, vector<duckdb::Value> &inputs, vector<duckdb::LogicalType> &arguments) {
    auto ring_type = LinearRegressionRingElement();
    function.return_type = ring_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);

}

duckdb::AggregateFunction GetToRingFunction() {
    auto arg_types = dukcbd::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE) // features
    };

    return duckdb::AggregateFunction(
        "to_ring",  // name 
        arg_types,  // argument types
        LinearRegressionRingElement, // return type
        duckdb::AggregateFunction::StateSize<ToRingState>, // state size
        duckdb::AggregateFunction::StateInitialize<ToRingState, ToRingFunction>, // state initialize
        ToRingUpdate, // update
        ToRingCombine, // combine
        ToRingFinalize, // finalize
        nullptr, // simple update
        ToRingBind, // bind
        duckdb::AggregateFunction::StateDestroy<ToRingState>, // state destroy
    );
}

void ToRing::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    
}

} // namespace quackml