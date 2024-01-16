// Testing conversion of tasks to DB calls
// Implementation of sum function, uses call to a sub database to compute operations

#include "functions/calls.hpp"

#include <iostream>
using std::cout;
using std::endl;

namespace quackml {

struct SumCallsOperation {
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
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);

        con.Query("CREATE TABLE t (i INTEGER);");
        con.Query("INSERT INTO t VALUES (" + std::to_string(state.sum) + "), (" + std::to_string(input) + ");");
        state.sum = con.Query("SELECT SUM(i) FROM t;")->GetValue<int>(0, 0);
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

duckdb::AggregateFunction GetSumCallFunction() {
    return duckdb::AggregateFunction::UnaryAggregate<SumCallsState, int, int, SumCallsOperation>(
        duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER);
}

void SumCalls::RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog) {
    duckdb::AggregateFunctionSet my_sum("sum_calls");
    my_sum.AddFunction(GetSumCallFunction());
    duckdb::CreateAggregateFunctionInfo my_sum_info(my_sum);
    catalog.CreateFunction(*conn.context, my_sum_info);
}

} // namespace quackml