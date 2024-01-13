// UDAF to compute sum and count of a relation

#include "functions/sum_count.hpp"

#include <map>

namespace quackml {

struct SumCountFunction {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.sum = 0;
        state.count = 0;
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        return;
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

static void SumCountUpdate(duckdb::Vector inputs[], duckdb::AggregateInputData &, idx_t input_count, duckdb::Vector &state_vector, idx_t count) {
    auto &input = inputs[0];
    duckdb::UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	duckdb::UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

    auto states = (SumCountState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
            auto &state = *states[sdata.sel->get_index(i)];
            // Cast input to double
            auto value = duckdb::UnifiedVectorFormat::GetData<double>(input_data);
            state.sum += value[input_data.sel->get_index(i)];
            state.count++;
        }
    }
}

static void SumCountCombine(duckdb::Vector &state_vector, duckdb::Vector &combined, duckdb::AggregateInputData &, idx_t count) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states_ptr = (SumCountState **)sdata.data;
    auto combined_ptr = duckdb::FlatVector::GetData<SumCountState *>(combined);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states_ptr[sdata.sel->get_index(i)];
        combined_ptr[i]->sum += state.sum;
        combined_ptr[i]->count += state.count;
    }
}

static void SumCountFinalize(duckdb::Vector &state_vector, duckdb::AggregateInputData &, duckdb::Vector &result, idx_t count, idx_t offset) {
    duckdb::UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (SumCountState **)sdata.data;
    auto &mask = duckdb::FlatVector::Validity(result);
    auto old_len = duckdb::ListVector::GetListSize(result);

    for (idx_t i = 0; i < count; i++) {
        const auto rid = i + offset;
        auto &state = *states[sdata.sel->get_index(i)];

        duckdb::Value sum_value = duckdb::Value::CreateValue(state.sum);
        duckdb::Value count_value = duckdb::Value::CreateValue(state.count);
        auto sum_pair = duckdb::Value::STRUCT({std::make_pair("key", "sum"), std::make_pair("value", sum_value)});
        auto count_pair = duckdb::Value::STRUCT({std::make_pair("key", "count"), std::make_pair("value", count_value)});
        duckdb::ListVector::PushBack(result, sum_pair);
        duckdb::ListVector::PushBack(result, count_pair);

        auto list_struct_data = duckdb::ListVector::GetData(result);
        list_struct_data[rid].length = duckdb::ListVector::GetListSize(result) - old_len;
        list_struct_data[rid].offset = old_len;
        old_len += list_struct_data[rid].length;
    }
    result.Verify(count);
}

duckdb::unique_ptr<duckdb::FunctionData> SumCountBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
    auto struct_type = duckdb::LogicalType::MAP(duckdb::LogicalType::VARCHAR, duckdb::LogicalType::DOUBLE);
    function.return_type = struct_type;
    return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
}

duckdb::AggregateFunction GetSumCountFunction() {
    using STATE_TYPE = SumCountState;

    return duckdb::AggregateFunction(
        "sum_count",                                                                // name
        {duckdb::LogicalType::DOUBLE},                                              // argument types
        duckdb::LogicalTypeId::MAP,                                                 // return type
        duckdb::AggregateFunction::StateSize<STATE_TYPE>,                           // state size
        duckdb::AggregateFunction::StateInitialize<STATE_TYPE, SumCountFunction>,   // initialize
        SumCountUpdate,                                                             // update
        SumCountCombine,                                                            // combine
        SumCountFinalize,                                                           // finalize
        nullptr,                                                                    // simple update 
        SumCountBind,                                                               // bind
        duckdb::AggregateFunction::StateDestroy<STATE_TYPE, SumCountFunction>       // destroy
    );
}

void SumCount::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet sum_count("sum_count");
    sum_count.AddFunction(GetSumCountFunction());
    duckdb::CreateAggregateFunctionInfo info(sum_count);
    catalog.CreateFunction(*conn.context, info);
}

} // namespace quackml