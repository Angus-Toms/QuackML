#define DUCKDB_EXTENSION_MAIN

#include "quackml_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include "functions/sum.hpp"
#include "functions/sum_count.hpp"
#include "functions/linear_reg.hpp"
#include "functions/to_ring.hpp"
#include "functions/linear_reg_ring.hpp"
#include "types/linear_reg_ring.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

#include <iostream>
#include <cmath>
#include <string>

namespace duckdb {

void get_rmse(DuckDB &db, std::string fname) {
    Connection con(db);
    con.Query("CREATE TABLE tbl AS SELECT * FROM read_csv_auto('" + fname + "', header=True, delim='\t', columns={'features': 'DOUBLE[]', 'label': 'DOUBLE'});");
    auto weights = con.Query("SELECT linear_regression(features, label, 0) FROM tbl;")->GetValue(0, 0);
    std::vector<double> weights_vector;
    for (auto &child : duckdb::ListValue::GetChildren(weights)) {
        weights_vector.push_back(child.GetValue<double>());
    }

    // Get test set features and labels 
    auto query = con.Query("SELECT * FROM tbl USING SAMPLE 20 PERCENT (bernoulli);");
    auto features = query->GetValue(0, 0);

    double sum_error = 0;
    for (idx_t i = 0; i < query->RowCount(); i++) {
        auto features = query->GetValue(0, i);
        auto label = query->GetValue(1, i);
        std::vector<double> feature_vector;
        for (auto &child : duckdb::ListValue::GetChildren(features)) {
            feature_vector.push_back(child.GetValue<double>());
        }

        // Set prediction to bias
        auto y_prediction = weights_vector[24];
        // Add weighted pairs 
        for (idx_t j = 0; j < 24; j++) {
            y_prediction += weights_vector[j] * feature_vector[j];
        }
        // Add squared error 
        sum_error += pow(label.GetValue<double>() - y_prediction, 2);
    }

    std::cout << fname << ":\n";
    std::cout << "Root Mean Squared Error: " << sqrt(sum_error / query->RowCount()) << "\n\n";

    con.Query("DROP TABLE tbl;");
}

void test_factorised_comp(DuckDB &db, std::string fname, idx_t n) {
    Connection con(db);
    auto lift_duration = 0;
    auto train_duration = 0;

    // Lift n input relations to ring elements
    for (idx_t i = 1; i < n+1; i++) {
        auto relation_fname = fname + std::to_string(i-1) + ".tsv";
        con.Query("CREATE TABLE tbl_" + std::to_string(i) + " AS SELECT * FROM read_csv_auto('" + relation_fname + "', header=True, delim='\t', columns={'features': 'DOUBLE[]', 'id': 'INTEGER'});");
        auto start_time = std::chrono::high_resolution_clock::now();
        con.Query("CREATE TABLE r" + std::to_string(i) + " AS SELECT id, to_ring(features) AS ring FROM tbl_" + std::to_string(i) + " GROUP BY id;");
        auto end_time = std::chrono::high_resolution_clock::now();
        lift_duration += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }

    // Train unfactorised linear regression model
    auto join_query = R"(
        CREATE TABLE joined AS SELECT 
        flatten([tbl_1.features[2:5], tbl_2.features, tbl_3.features, tbl_4.features, tbl_5.features, tbl_6.features, tbl_7.features, tbl_8.features, tbl_9.features, tbl_10.features]) AS features,
        tbl_1.features[1] AS labels
        FROM tbl_1
        JOIN tbl_2 ON tbl_1.id = tbl_2.id
        JOIN tbl_3 ON tbl_1.id = tbl_3.id
        JOIN tbl_4 ON tbl_1.id = tbl_4.id
        JOIN tbl_5 ON tbl_1.id = tbl_5.id
        JOIN tbl_6 ON tbl_1.id = tbl_6.id
        JOIN tbl_7 ON tbl_1.id = tbl_7.id
        JOIN tbl_8 ON tbl_1.id = tbl_8.id
        JOIN tbl_9 ON tbl_1.id = tbl_9.id
        JOIN tbl_10 ON tbl_1.id = tbl_10.id;
    )";
    // std::cout << join_query << "\n";
    auto join_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Join query:\n";
    con.Query(join_query)->Print();
    con.Query("SELECT * FROM joined;");
    auto join_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Join duration: " << std::chrono::duration_cast<std::chrono::microseconds>(join_end_time - join_start_time).count() << "micros\n";
    auto train_start_time = std::chrono::high_resolution_clock::now();
    auto train_query = R"(
        SELECT linear_regression(features, labels, 0) 
        FROM joined;
    )";
    con.Query(train_query)->Print();
    auto train_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Train duration: " << std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count() << "micros\n";

    // Train factorised linear regression model on ring elements
    std::cout << "Factorised linear regression model:\n";
    auto factorised_train_start = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        SELECT linear_regression_ring([r1.ring, r2.ring, r3.ring, r4.ring, r5.ring, r6.ring, r7.ring, r8.ring, r9.ring, r10.ring], 0) 
        FROM r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 
        WHERE r1.id = r2.id 
        AND r1.id = r3.id 
        AND r1.id = r4.id
        AND r1.id = r5.id
        AND r1.id = r6.id
        AND r1.id = r7.id
        AND r1.id = r8.id
        AND r1.id = r9.id
        AND r1.id = r10.id
    )")->Print();
    auto factorised_train_end = std::chrono::high_resolution_clock::now();
    train_duration = std::chrono::duration_cast<std::chrono::microseconds>(factorised_train_end - factorised_train_start).count();
    std::cout << "Lift duration: " << lift_duration << "micros\n";
    std::cout << "Train duration: " << train_duration << "micros\n\n";
}


void run_quackml_tests(DuckDB &db) {
    std::cout << "<=========== Running QuackML tests ===========>\n";

    Connection con(db);

    // Accuracy tests 
    // get_rmse(db, "test/quackml/datasets/test_8.tsv");
    // get_rmse(db, "test/quackml/datasets/test_9.tsv");
    // get_rmse(db, "test/quackml/datasets/test_10.tsv");
    // get_rmse(db, "test/quackml/datasets/test_11.tsv");
    // get_rmse(db, "test/quackml/datasets/test_12.tsv");
    // get_rmse(db, "test/quackml/datasets/test_13.tsv");
    // get_rmse(db, "test/quackml/datasets/test_14.tsv");
    // get_rmse(db, "test/quackml/datasets/test_15.tsv");
    // get_rmse(db, "test/quackml/datasets/test_16.tsv");
    // get_rmse(db, "test/quackml/datasets/test_17.tsv");
    // get_rmse(db, "test/quackml/datasets/test_18.tsv");
    // get_rmse(db, "test/quackml/datasets/test_19.tsv");
    // get_rmse(db, "test/quackml/datasets/test_20.tsv");
    
   
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << "ms\n";

    // Flight test set
    con.Query("CREATE TABLE airports AS SELECT * FROM read_csv_auto('test/quackml/flights/airports_clean.csv');");
    con.Query("CREATE TABLE flights AS SELECT * FROM read_csv_auto('test/quackml/flights/flights_clean.csv');");
    con.Query("CREATE TABLE weather AS SELECT * FROM read_csv_auto('test/quackml/flights/weather_clean.csv');");
    con.Query("CREATE TABLE planes AS SELECT * FROM read_csv_auto('test/quackml/flights/planes_clean.csv');");
    con.Query("CREATE TABLE airlines AS SELECT * FROM read_csv_auto('test/quackml/flights/airlines_clean.csv');");

    // Construct join 
    // START HERE: There are nulls somewhere in this dataset, remove!!!!
    // dest_mean_delay contains nulls (7k)
    // START HERE: Add encoded timezones from airports
    auto join_start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        CREATE TABLE joined AS SELECT 
        [
            l.name_mean_delay, 
            f.month, f.day, f.hour, f.minute, f.distance, f.dep_delay_int, f.arr_delay_int, f.air_time_int, f.dest_mean_delay,
            a.lat, a.lon, a.alt, a.tz, a.Asia_Chongqing, a.Pacific_Honolulu, a.America_Chicago, a.NA, a.America_Los_Angeles, a.America_Vancouver, a.America_Anchorage, a.America_Denver, a.America_New_York, a.America_Phoenix,
            w.precip, w.visib, w.temp_double, w.dewp_double, w.humid_double, w.wind_speed_int, w.pressure_double,
            p.engines, p.seats, p.year_int, p.model_mean_delay, p.manufacturer_mean_delay, p.fixed_wing_multi, p.fixed_wing_single, p.rotorcraft, p.turbo_jet, p.turbo_prop, p.turbo_shaft, p.turbo_fan, p.reciprocating, p.four_cycle
        ] AS features,
        f.arr_delay_int AS labels 
        FROM flights f
        JOIN airports a ON f.origin = a.faa 
        JOIN weather w ON f.origin = w.origin AND f.time_hour = w.time_hour 
        JOIN planes p ON f.tailnum = p.tailnum
        JOIN airlines l ON f.carrier = l.carrier;
    )")->Print();
    auto join_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Join duration: " << std::chrono::duration_cast<std::chrono::microseconds>(join_end_time - join_start_time).count() << " micros\n";

    // Train model 
    auto train_start_time = std::chrono::high_resolution_clock::now();
    auto query = R"(
        SELECT linear_regression(features, labels, 0) 
        FROM joined;
    )";
    auto query_run = con.Query(query);
    query_run->Print();
    auto train_end_time = std::chrono::high_resolution_clock::now();
    auto weights = query_run->GetValue(0, 0);
    std::cout << "Train duration: " << std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count() << " micros\n";
    
    
    std::vector<double> weights_vector;
    for (auto &child : duckdb::ListValue::GetChildren(weights)) {
        weights_vector.push_back(child.GetValue<double>());
    }

    // Get RMSE of loosely-integrated model 
    auto sample_query = con.Query("SELECT * FROM joined USING SAMPLE 20 PERCENT (bernoulli);");
    auto features = sample_query->GetValue(0, 0);
    double sum_error = 0;

    for (idx_t i = 0; i < sample_query->RowCount(); i++) {
        auto features = sample_query->GetValue(0, i);
        auto label = sample_query->GetValue(1, i);

        std::vector<double> feature_vector;
        for (auto &child : duckdb::ListValue::GetChildren(features)) {
            feature_vector.push_back(child.GetValue<double>());
        }

        auto prediction = 0;
        // Add weighted pairs 
        for (idx_t j = 0; j < weights_vector.size(); j++) {
            prediction += weights_vector[j] * feature_vector[j];
        }

        // Add squared error 
        sum_error += pow(label.GetValue<double>() - prediction, 2);
    }
    std::cout << "Loosely-integrated RMSE: " << sqrt(sum_error / sample_query->RowCount()) << "\n\n";


    // Factorised model 
    // Create ring elements
    auto lift_start_time = std::chrono::high_resolution_clock::now();
    auto flight_start = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        CREATE TABLE flight_ring AS
        SELECT f.origin, f.carrier, to_ring([f.arr_delay_int, f.month, f.day, f.hour, f.minute, f.distance, f.dep_delay_int, f.air_time_int, f.dest_mean_delay, p.engines, p.seats, p.year_int, p.model_mean_delay, p.manufacturer_mean_delay, p.fixed_wing_multi, p.fixed_wing_single, p.rotorcraft, p.turbo_jet, p.turbo_prop, p.turbo_shaft, p.turbo_fan, p.reciprocating, p.four_cycle, w.precip, w.visib, w.temp_double, w.dewp_double, w.humid_double, w.wind_speed_int, w.pressure_double]) ring
        FROM flights f
        JOIN planes p ON f.tailnum = p.tailnum
        JOIN weather w ON f.origin = w.origin AND f.time_hour = w.time_hour
        GROUP BY f.origin, f.carrier;
    )");
    auto flight_end = std::chrono::high_resolution_clock::now();
    std::cout << "Flight ring duration: " << std::chrono::duration_cast<std::chrono::microseconds>(flight_end - flight_start).count() << "micros\n";
    auto airport_start = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        CREATE TABLE airport_ring AS
        SELECT faa, to_ring([lat, lon, alt, tz, Asia_Chongqing, Pacific_Honolulu, America_Chicago, NA, America_Los_Angeles, America_Vancouver, America_Anchorage, America_Denver, America_New_York, America_Phoenix]) ring
        FROM airports GROUP BY faa;
    )");
    auto airport_end = std::chrono::high_resolution_clock::now();
    std::cout << "Airport ring duration: " << std::chrono::duration_cast<std::chrono::microseconds>(airport_end - airport_start).count() << "micros\n";
    auto airline_start = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        CREATE TABLE airline_ring AS
        SELECT carrier, to_ring([name_mean_delay]) ring
        FROM airlines GROUP BY carrier;
    )");
    auto airline_end = std::chrono::high_resolution_clock::now();
    std::cout << "Airline ring duration: " << std::chrono::duration_cast<std::chrono::microseconds>(airline_end - airline_start).count() << "micros\n";
    auto lift_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Lift duration: " << std::chrono::duration_cast<std::chrono::microseconds>(lift_end_time - lift_start_time).count() << "micros\n";

    // Train factorised model
    auto factorised_train_start = std::chrono::high_resolution_clock::now();
    auto result = con.Query(R"(
        SELECT linear_regression_ring([f.ring, a.ring, l.ring], 0) 
        FROM flight_ring f, airport_ring a, airline_ring l
        WHERE f.origin = a.faa 
        AND f.carrier = l.carrier;
    )");
    auto factorised_train_end = std::chrono::high_resolution_clock::now();
    result->Print();
    
    // Get RMSE of factorised model
    auto factorised_weights = result->GetValue(0, 0);
    std::vector<double> factorised_weights_v;
    for (auto &child : duckdb::ListValue::GetChildren(factorised_weights)) {
        factorised_weights_v.push_back(child.GetValue<double>());
    }

    double factorised_sum_error = 0;
    for (idx_t i = 0; i < sample_query->RowCount(); i++) {
        auto features = sample_query->GetValue(0, i);
        auto label = sample_query->GetValue(1, i);
        std::vector<double> feature_vector;
        for (auto &child : duckdb::ListValue::GetChildren(features)) {
            feature_vector.push_back(child.GetValue<double>());
        }

        auto prediction = 0;
        // Add weighted pairs 
        for (idx_t j = 0; j < factorised_weights_v.size(); j++) {
            prediction += factorised_weights_v[j] * feature_vector[j];
        }
        // Add squared error 
        factorised_sum_error += pow(label.GetValue<double>() - prediction, 2);
    }
    std::cout << "Factorised RMSE: " << sqrt(factorised_sum_error / sample_query->RowCount()) << "\n\n";

    // Learning over cartesian products 
    // test_factorised_comp(db, "test/quackml/datasets/factorised_", 10);

    std::cout << "<========== QuackML tests complete ==========>\n";
}


void QuackmlExtension::Load(DuckDB &db) {
	Connection con(db);
    con.BeginTransaction();
    auto &catalog = Catalog::GetSystemCatalog(*con.context);

    quackml::Sum::RegisterFunction(con, catalog);
    quackml::SumCount::RegisterFunction(con, catalog);
    quackml::LinearRegression::RegisterFunction(con, catalog);
    quackml::ToRing::RegisterFunction(con, catalog);
    quackml::LinearRegressionRing::RegisterFunction(con, catalog);

    con.Commit();

    //run_quackml_tests(db);
}

std::string QuackmlExtension::Name() {
	return "QuackML";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void quackml_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::QuackmlExtension>();
}

DUCKDB_EXTENSION_API const char *quackml_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
