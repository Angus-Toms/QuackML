// Sum count UDAF implementation built with different constructor

#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/vector.hpp"
#include <string>
#include <map>

using std::string;
using std::map;
using namespace duckdb;

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
    static void Destroy(STATE &state, AggregateInputData &aggr_input_data) {
        if (state.result) {
            delete state.result;
        }
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

template <class MAP_TYPE>
static void SumCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    auto &input = inputs[0];
    UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

    // Then OP::template HistogramUpdate<T, MAP_TYPE>(sdata, input_data, count) called
    auto states = (SumCountState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            // Result map not yet initialized
            if (!state.result) {
                state.result = new MAP_TYPE();
            }
            // Cast input to
            auto value = UnifiedVectorFormat::GetData<double>(input_data);
            //(*state.result)[value[input_data.sel->get_index(i)]]++;
            (*state.result)["sum"] += (*value);
            (*state.result)["count"]++;
        }
    }
}

template <class MAP_TYPE>
static void SumCountCombine(Vector &state_vector, Vector &combined, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (SumCountState **)sdata.data;

    auto combined_ptr = FlatVector::GetData<SumCountState *>(combined);

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
        //for (auto &entry : *state.result) {
        //    (*combined_ptr[i]->result)[entry.first] += entry.second;
        //}

    }
}

template <class MAP_TYPE>
static void SumCountFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (SumCountState **)sdata.data;
    auto &mask = FlatVector::Validity(result);
    auto old_len = ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];
        if (!state.result) {
            mask.SetInvalid(rid);
            continue;
        }

        //for (auto &entry : *state.result) {
        //    Value bucket_value = Value::CreateValue(entry.first);
        //    auto count_value = Value::CreateValue(entry.second);
        //    auto struct_value = Value::STRUCT({std::make_pair("key", bucket_value), std::make_pair("value", count_value)});
        //    ListVector::PushBack(result, struct_value);
        //}

        Value sum_value = Value::CreateValue((*state.result)["sum"]);
        Value count_value = Value::CreateValue((*state.result)["count"]);
        auto sum_pair = Value::STRUCT({std::make_pair("key", "sum"), std::make_pair("value", sum_value)});
        auto count_pair = Value::STRUCT({std::make_pair("key", "count"), std::make_pair("value", count_value)});
        ListVector::PushBack(result, sum_pair);
        ListVector::PushBack(result, count_pair);

        auto list_struct_data = ListVector::GetData(result);
        list_struct_data[rid].length = ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

unique_ptr<FunctionData> SumCountBind(ClientContext &context, AggregateFunction &function, vector<unique_ptr<Expression>> &arguments) {
    auto struct_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE);
    function.return_type = struct_type;
    return make_uniq<VariableReturnBindData>(function.return_type);
}

AggregateFunction GetSumCountFunction() {
    using MAP_TYPE = map<string, double>;
    using STATE_TYPE = SumCountState;

    return AggregateFunction(
        "sum_count",                                                                // name
        {LogicalType::DOUBLE},                                                      // argument types
        LogicalTypeId::MAP,                                                         // return type
        AggregateFunction::StateSize<STATE_TYPE>,                                   // state size
        AggregateFunction::StateInitialize<STATE_TYPE, SumCountFunction>,           // initialize
        SumCountUpdate<MAP_TYPE>,                                                   // update
        SumCountCombine<MAP_TYPE>,                                                  // combine
        SumCountFinalize<MAP_TYPE>,                                                 // finalize
        nullptr,                                                                    // simple update 
        SumCountBind,                                                               // bind
        AggregateFunction::StateDestroy<STATE_TYPE, SumCountFunction>               // destroy
    );
}

void RegisterSumCountFunction(Connection &conn, Catalog &catalog) {
    AggregateFunctionSet sum_count_2("sum_count_2");
    sum_count_2.AddFunction(GetSumCountFunction());
    CreateAggregateFunctionInfo info(sum_count_2);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace duckdb