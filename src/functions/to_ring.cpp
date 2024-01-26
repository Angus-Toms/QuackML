// This aggregate function converts an arbitrary number of fields in a relation
// to a single linear regression ring element. These elements can then be 
// combined by the linear regression compute function.

#include "functions/to_ring.hpp"

namespace quackml {

struct ToRingState {
    LinearRegressionRingElement *ringElement;
};

struct ToRingFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.ringElement = nullptr;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        if (state.ringElement) {
            delete state.ringElement;
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

    for (idx_t i = 0; i < count; i++) {
        if (features_data.validity.RowIsValid(features_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            auto feature_vector = duckdb::ListValue::GetChildren(features.GetValue(i));
            if (!state.ringElement) {
                state.ringElement = new LinearRegressionRingElement(feature_vector.size());
            }
            
            // Convert feature_vector to std::vector<double>
            std::vector<double> feature_std_vector = std::vector<double>();
            for (idx_t j = 0; j < feature_vector.size(); j++) {
                feature_std_vector.push_back(feature_vector[j].GetValue<double>());
            }
            auto tuple_ring = LinearRegressionRingElement(feature_std_vector);
            auto ring_increment = *state.ringElement + tuple_ring;
            state.ringElement = &ring_increment;

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
        *(combined_ptr[i]->ringElement) = *(combined_ptr[i]->ringElement) + *state.ringElement;
    }
}

static void ToRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ToRingState **)sdata.data;

    LinearRegressionRingElement element;
    duckdb::ListVector::PushBack(result, element);
}

duckdb::unique_ptr<duckdb::FunctionData> ToRingBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto ring_type = LinearRegressionRingElement();
    function.return_type = ring_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

duckdb::AggregateFunction GetToRingFunction() {
    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE) // features
    };

    return duckdb::AggregateFunction(
        "to_ring",                                                                  // name 
        arg_types,                                                                  // argument types
        LinearRegressionRingElement(),                                                 // return type
        duckdb::AggregateFunction::StateSize<ToRingState>,                          // state size
        duckdb::AggregateFunction::StateInitialize<ToRingState, ToRingFunction>,    // state initialize
        ToRingUpdate,                                                               // update
        ToRingCombine,                                                              // combine
        ToRingFinalize,                                                             // finalize
        nullptr,                                                                    // simple update
        ToRingBind,                                                                 // bind
        duckdb::AggregateFunction::StateDestroy<ToRingState, ToRingFunction>        // state destroy
    );
}

void ToRing::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet to_ring("to_ring");
    to_ring.AddFunction(GetToRingFunction());
    duckdb::CreateAggregateFunctionInfo info(to_ring);
    catalog.CreateFunction(*conn.context, info);    
}

} // namespace quackml