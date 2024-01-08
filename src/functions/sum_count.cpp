// UDAF to find sum and count in single function

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

struct SumCountOperation {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.result = nullptr;
    }

    template <class STATE, class OP>
    static void Combine(const STATE &source, STATE &target, duckdb::AggregateInputData &) {
        // source is const so we have to use .at() instead of []
        if (!source.result) {
            return;
        }
        
        if (!target.result) {
            target.result = new map<string, double>();
        }
        
        (*target.result)["sum"] += (*source.result).at("sum");
        (*target.result)["count"] += (*source.result).at("count");
    }

    template <class T, class STATE>
    static void Finalize(STATE &state, T &target, duckdb::AggregateFinalizeData &finalize_data) {
        if (!state.result) {
            finalize_data.ReturnNull();
            return;
        } 
        target = *state.result;
    }

    template <class INPUT_TYPE, class STATE, class OP>
    static void Operation(STATE &state, const INPUT_TYPE &input, duckdb::AggregateUnaryInput &unary_input) {
        if (!state.result) {
            state.result = new map<string, double>();
        }
        (*state.result)["sum"] += input;
        (*state.result)["count"]++;
    }

    template <class INPUT_TYPE, class STATE, class OP>
    static void ConstantOperation(STATE &state, const INPUT_TYPE &input, duckdb::AggregateUnaryInput &unary_input, idx_t count) {
        for (idx_t i = 0; i < count; i++) {
            Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
        }
    }

    template <class STATE>
    static void Destroy(STATE &state, duckdb::AggregateInputData &aggr_input_data) {
        delete state.result;
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

duckdb::AggregateFunction GetSumCountFunction() {
    return duckdb::AggregateFunction::UnaryAggregate<SumCountState, double, map<string, double>, SumCountOperation>(
        duckdb::LogicalType::DOUBLE, duckdb::LogicalTypeId::MAP
    );
}

void RegisterSumCountFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet sum_count("sum_count");
    sum_count.AddFunction(GetSumCountFunction());
    duckdb::CreateAggregateFunctionInfo sum_count_info(sum_count);
    catalog.CreateFunction(*conn.context, sum_count_info);
}

} // namespace ml