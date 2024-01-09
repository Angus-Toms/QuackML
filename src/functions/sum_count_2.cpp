// Sum count UDAF implementation built with different constructor

#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include <string>
#include <map>

using std::string;
using std::map;

namespace ml {

struct SumCountState {
    map<string, double> *result;
};

struct SumCountFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.result = nullptr;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        if (state.result) {
            delete state.result;
        }
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

template <class MAP_TYPE>
static void SumCountUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &input = inputs[0];
    duckdb::UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	duckdb::UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

    // Then OP::template HistogramUpdate<T, MAP_TYPE>(sdata, input_data, count) called
    auto states = (SumCountState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            if (!state.result) {
                state.result = new MAP_TYPE();
            }
            auto value = duckdb::UnifiedVectorFormat::GetData<double>(input_data);
            (*state.result)["sum"] += value[input_data.sel->get_index(i)];
            (*state.result)["count"]++;
        }
    }
}

template <class MAP_TYPE>
static void SumCountCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (SumCountState **)sdata.data;

    auto combined_ptr = duckdb::FlatVector::GetData<SumCountState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        if (!state.result) {
            continue;
        }
        if (!combined_ptr[i]->result) {
            combined_ptr[i]->result = new MAP_TYPE();
        }
        (*combined_ptr[i]->result)["sum"] += (*state.result)["sum"];
        (*combined_ptr[i]->result)["count"] += (*state.result)["count"];
    }
}

template <class MAP_TYPE>
static void SumCountFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (SumCountState **)sdata.data;

    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];
        if (!state.result) {
            mask.SetInvalid(rid);
            continue;
        }

        // OP::template HistogramFinalize<T>(entry.first) called
        // Construct result struct?
        duckdb::Value sum_value = duckdb::Value::CreateValue((*state.result)["sum"]);
        duckdb::Value count_value = duckdb::Value::CreateValue((*state.result)["count"]);
        auto struct_value = 
            duckdb::Value::STRUCT({std::make_pair("sum", sum_value), std::make_pair("count", count_value)});
        duckdb::ListVector::PushBack(result, struct_value);

        // Unsure what this does?
        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

duckdb::AggregateFunction GetSumCountFunction() {
    using MAP_TYPE = map<string, double>;
    using STATE_TYPE = SumCountState;

    return duckdb::AggregateFunction(
        "sum_count",                                                                // name
        {duckdb::LogicalType::DOUBLE},                                              // argument types
        duckdb::LogicalTypeId::MAP,                                                 // return type
        duckdb::AggregateFunction::StateSize<STATE_TYPE>,                           // state size
        duckdb::AggregateFunction::StateInitialize<STATE_TYPE, SumCountFunction>,   // initialize
        SumCountUpdate<MAP_TYPE>,                                                   // update
        SumCountCombine<MAP_TYPE>,                                                  // combine
        SumCountFinalize<MAP_TYPE>,                                                 // finalize
        nullptr,                                                                    // simple update 
        nullptr,                                                                    // bind (histogram does actually implement this)
        duckdb::AggregateFunction::StateDestroy<STATE_TYPE, SumCountFunction>       // destroy
    );
}

void RegisterSumCountFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet sum_count_2("sum_count_2");
    sum_count_2.AddFunction(GetSumCountFunction());
    duckdb::CreateAggregateFunctionInfo info(sum_count_2);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace ml