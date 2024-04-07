// This aggregate function converts an arbitrary number of fields in a relation
// to a single linear regression ring element. These elements can then be 
// combined by the linear regression compute function.
#include "functions/to_ring.hpp"

namespace quackml {

struct ToRingState {
    LinRegRing *ring;
};


struct ToRingFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.ring = nullptr;
    }
    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {

    }
    static bool IgnoreNull() {
        return true;
    }
};

static void ToRingUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &features = inputs[0];
    auto &labels = inputs[1];
    duckdb::UnifiedVectorFormat features_data;
    features.ToUnifiedFormat(count, features_data);

    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ToRingState **)sdata.data;

    std::map<ToRingState*, std::vector<duckdb::vector<duckdb::Value>>*> state_features;
    for (idx_t i = 0; i < count; i++) {
        // Get state address for this tuple 
        ToRingState* state_ptr = states[sdata.sel->get_index(i)];
        // Get feature vector address for this tuple 
        auto feature_vector = duckdb::ListValue::GetChildren(features.GetValue(i));
        // Initialize state's feature vector if it doesn't exist
        if (state_features.find(state_ptr) == state_features.end()) {
            state_features[state_ptr] = new std::vector<duckdb::vector<duckdb::Value>>();
        }
        // Add feature vector to map of <states, [feature_vector]> 
        state_features[state_ptr]->push_back(feature_vector);
    }

    // Iterate through map of <states, [feature_vector]> 
    for (auto const& pair : state_features) {
        // Create ringElement from feature vectors for each state
        auto state = pair.first;
        auto observations = pair.second;
        auto ring = new LinRegRing(*observations);
        if (!state->ring) {
            state->ring = ring;
        } else {
            // If state already has ring, add new ring to it
            *state->ring = *state->ring + *ring;
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
        // State ring not instantiated, skip
        if (!state.ring) {
            continue;
        }

        // Combined ring not instantiated, copy state ring 
        if (!combined_ptr[i]->ring) {
            combined_ptr[i]->ring = new LinRegRing(*state.ring);
        
        // State and combined exist, add rings
        } else {
            *combined_ptr[i]->ring = *combined_ptr[i]->ring + *state.ring;
        }
    }
}

static void ToRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t in_count, idx_t offset) {    
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(in_count, sdata);
    auto states = (ToRingState **)sdata.data;
    auto old_len = duckdb::ListVector::GetListSize(result);
    for (idx_t i = 0; i < in_count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];

        // Wrap count
        auto count = state.ring->count;
        auto count_value = duckdb::Value::LIST({duckdb::Value::DOUBLE(count)});

        // Wrap sums
        auto sums = state.ring->sums;
        duckdb::vector<duckdb::Value> sum_values;
        for (auto &sum : sums) {
            sum_values.push_back(duckdb::Value::DOUBLE(sum));
        }

        // Wrap covar
        auto covar = state.ring->covar;
        duckdb::vector<duckdb::Value> covar_values;
        for (auto &row : covar) {
            duckdb::vector<duckdb::Value> row_values;
            for (auto &val : row) {
                row_values.push_back(duckdb::Value::DOUBLE(val));
            }
            covar_values.push_back(duckdb::Value::LIST(row_values));
        }

        // Add to results 
        duckdb::ListVector::PushBack(result, duckdb::Value::LIST({count_value}));
        duckdb::ListVector::PushBack(result, duckdb::Value::LIST({duckdb::Value::LIST(sum_values)}));
        duckdb::ListVector::PushBack(result, duckdb::Value::LIST(covar_values));

        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(in_count);
}

static void FastToRingFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t in_count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(in_count, sdata);
    auto states = (ToRingState **)sdata.data;
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < in_count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];

        // Add d to result 
        auto d = state.ring->sums.size();
        duckdb::ListVector::PushBack(result, duckdb::Value::DOUBLE(d));

        // Add count to results 
        auto count = state.ring->count;
        duckdb::ListVector::PushBack(result, duckdb::Value::DOUBLE(count));

        // Add sums to results
        for (auto &sum : state.ring->sums) {
            duckdb::ListVector::PushBack(result, duckdb::Value::DOUBLE(sum));
        }

        // Add covar 
        for (auto &row : state.ring->covar) {
            for (auto &val : row) {
                duckdb::ListVector::PushBack(result, duckdb::Value::DOUBLE(val));
            }
        }

        // Set length 
        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;

    }
    result.Verify(in_count);
}

duckdb::unique_ptr<duckdb::FunctionData> ToRingBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    function.return_type = duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

duckdb::AggregateFunction GetToRingFunction() {
    auto arg_types = duckdb::vector<duckdb::LogicalType>{
        duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE) // features
    };

    return duckdb::AggregateFunction(
        "to_ring",                                                                  // name 
        arg_types,                                                                  // argument types
        duckdb::LogicalTypeId::LIST,                                                // return type
        duckdb::AggregateFunction::StateSize<ToRingState>,                          // state size
        duckdb::AggregateFunction::StateInitialize<ToRingState, ToRingFunction>,    // state initialize
        ToRingUpdate,                                                               // update
        ToRingCombine,                                                              // combine
        FastToRingFinalize,                                                             // finalize
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
