// Sum count UDAF implementation built with different constructor

#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/vector.hpp"

#include <map>

using namespace duckdb;

namespace ml {

struct SumCountState {
    double sum;
    double count;
};

struct SumCountFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.sum = 0;
        state.count = 0;
    }

    template <class STATE>
    static void Destroy(STATE &state, AggregateInputData &aggr_input_data) {
        return;
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

static void SumCountUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
    auto &input = inputs[0];
    UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

    auto states = (SumCountState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            // Cast input to double
            auto value = UnifiedVectorFormat::GetData<double>(input_data);
            state.sum += value[input_data.sel->get_index(i)];
            state.count++;
        }
    }
}

static void SumCountCombine(Vector &state_vector, Vector &combined, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (SumCountState **)sdata.data;
    auto combined_ptr = FlatVector::GetData<SumCountState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->sum += state.sum;
        combined_ptr[i]->count += state.count;
    }
}

static void SumCountFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (SumCountState **)sdata.data;
    auto &mask = FlatVector::Validity(result);
    auto old_len = ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];

        Value sum_value = Value::CreateValue(state.sum);
        Value count_value = Value::CreateValue(state.count);
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
    using STATE_TYPE = SumCountState;

    return AggregateFunction(
        "sum_count",                                                                // name
        {LogicalType::DOUBLE},                                                      // argument types
        LogicalTypeId::MAP,                                                         // return type
        AggregateFunction::StateSize<STATE_TYPE>,                                   // state size
        AggregateFunction::StateInitialize<STATE_TYPE, SumCountFunction>,           // initialize
        SumCountUpdate,                                                             // update
        SumCountCombine,                                                            // combine
        SumCountFinalize,                                                           // finalize
        nullptr,                                                                    // simple update 
        SumCountBind,                                                               // bind
        AggregateFunction::StateDestroy<STATE_TYPE, SumCountFunction>               // destroy
    );
}

void RegisterSumCountFunction(Connection &conn, Catalog &catalog) {
    AggregateFunctionSet sum_count("sum_count");
    sum_count.AddFunction(GetSumCountFunction());
    CreateAggregateFunctionInfo info(sum_count);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace duckdb