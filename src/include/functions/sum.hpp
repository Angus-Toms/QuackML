// sum.hpp

#ifndef SUM_HPP
#define SUM_HPP

#pragma once
#include "duckdb.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

namespace quackml {

struct SumState {
    double sum;
};

struct SumOperation;

struct Sum {
    static void RegisterFunction(duckdb::Connection &conn, duckdb::Catalog &catalog);
};

} // namespace quackml

#endif // SUM_HPP