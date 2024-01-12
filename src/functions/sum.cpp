// Redefinition of sum aggregate function
// Just to get the hang of UDAFs

#include "functions/sum.hpp"

namespace quackml {

struct SumOperation {
    template <class STATE>
    static void Initialize(STATE &state) {
        state.sum = 0;
    }

    template <class STATE, class OP>
    static void Combine(const STATE &source, STATE &target, duckdb::AggregateInputData &) {
        target.sum += source.sum;
    }

    template <class T, class STATE>
    static void Finalize(STATE &state, T &target, duckdb::AggregateFinalizeData &finalize_data) {
        target = state.sum;
    }

    template <class INPUT_TYPE, class STATE, class OP>
    static void Operation(STATE &state, const INPUT_TYPE &input, duckdb::AggregateUnaryInput &unary_input) {
        state.sum += input; 
    }

    template <class INPUT_TYPE, class STATE, class OP>
    static void ConstantOperation(STATE &state, const INPUT_TYPE &input, duckdb::AggregateUnaryInput &unary_input, idx_t count) {
        for (idx_t i = 0; i < count; i++) {
            Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
        }
    }

    static bool IgnoreNull() { 
        return true; 
    }
};

duckdb::AggregateFunction GetSumFunction() {
    return duckdb::AggregateFunction::UnaryAggregate<SumState, double, double, SumOperation>(
        duckdb::LogicalType::DOUBLE, duckdb::LogicalType::DOUBLE);
}

void Sum::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet my_sum("my_sum");
    my_sum.AddFunction(GetSumFunction());
    duckdb::CreateAggregateFunctionInfo my_sum_info(my_sum);
    catalog.CreateFunction(*conn.context, my_sum_info);
}

} // namespace quackml