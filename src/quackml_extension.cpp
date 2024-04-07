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

void test_regression(duckdb::Connection &con) {
    con.Query("CREATE TABLE test AS SELECT * FROM read_csv('test/quackml/datasets/test_5.tsv', header=TRUE, delim='\t', columns={'features': 'DOUBLE[]', 'labels': 'DOUBLE'});");
    con.Query("SELECT linear_regression(features, labels, 0) FROM test;")->Print();
    con.Query("CREATE TABLE r AS SELECT to_ring(features) AS ring FROM test;");
    con.Query("SELECT linear_regression_ring([ring], 0) FROM r;")->Print();
}

void test_flights(duckdb::Connection &con) {
    // Flight test set
    con.Query("CREATE TABLE airports AS SELECT * FROM read_csv_auto('test/quackml/flights/airports_clean.csv');");
    con.Query("CREATE TABLE flights AS SELECT * FROM read_csv_auto('test/quackml/flights/flights_clean.csv');");
    con.Query("CREATE TABLE weather AS SELECT * FROM read_csv_auto('test/quackml/flights/weather_clean.csv');");
    con.Query("CREATE TABLE planes AS SELECT * FROM read_csv_auto('test/quackml/flights/planes_clean.csv');");
    con.Query("CREATE TABLE airlines AS SELECT * FROM read_csv_auto('test/quackml/flights/airlines_clean.csv');");

    // Construct join 
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
        SELECT linear_regression_ring([f.ring, a.ring, l.ring], 0) 
        FROM (
            SELECT 
                f.origin, 
                f.carrier, 
                to_ring([f.arr_delay_int, f.month, f.day, f.hour, f.minute, f.distance, f.dep_delay_int, f.air_time_int, f.dest_mean_delay, p.engines, p.seats, p.year_int, p.model_mean_delay, p.manufacturer_mean_delay, p.fixed_wing_multi, p.fixed_wing_single, p.rotorcraft, p.turbo_jet, p.turbo_prop, p.turbo_shaft, p.turbo_fan, p.reciprocating, p.four_cycle, w.precip, w.visib, w.temp_double, w.dewp_double, w.humid_double, w.wind_speed_int, w.pressure_double]) AS ring
            FROM 
                flights f
            JOIN 
                planes p ON f.tailnum = p.tailnum
            JOIN 
                weather w ON f.origin = w.origin AND f.time_hour = w.time_hour
            GROUP BY 
                f.origin, f.carrier
        ) AS f,
        (
            SELECT 
                faa, 
                to_ring([lat, lon, alt, tz, Asia_Chongqing, Pacific_Honolulu, America_Chicago, NA, America_Los_Angeles, America_Vancouver, America_Anchorage, America_Denver, America_New_York, America_Phoenix]) AS ring
            FROM 
                airports 
            GROUP BY 
                faa
        ) AS a,
        (
            SELECT 
                carrier, 
                to_ring([name_mean_delay]) AS ring
            FROM 
                airlines 
            GROUP BY 
                carrier
        ) AS l
        WHERE 
            f.origin = a.faa 
            AND f.carrier = l.carrier
    )")->Print();
    auto factorised_train_end = std::chrono::high_resolution_clock::now();
    std::cout << "Factorised train duration: " << std::chrono::duration_cast<std::chrono::microseconds>(factorised_train_end - flight_start).count() << "micros\n";
    
    // Get RMSE of factorised model
    // auto factorised_weights = result->GetValue(0, 0);
    // std::vector<double> factorised_weights_v;
    // for (auto &child : duckdb::ListValue::GetChildren(factorised_weights)) {
    //     factorised_weights_v.push_back(child.GetValue<double>());
    // }

    // double factorised_sum_error = 0;
    // for (idx_t i = 0; i < sample_query->RowCount(); i++) {
    //     auto features = sample_query->GetValue(0, i);
    //     auto label = sample_query->GetValue(1, i);
    //     std::vector<double> feature_vector;
    //     for (auto &child : duckdb::ListValue::GetChildren(features)) {
    //         feature_vector.push_back(child.GetValue<double>());
    //     }

    //     auto prediction = 0;
    //     // Add weighted pairs 
    //     for (idx_t j = 0; j < factorised_weights_v.size(); j++) {
    //         prediction += factorised_weights_v[j] * feature_vector[j];
    //     }
    //     // Add squared error 
    //     factorised_sum_error += pow(label.GetValue<double>() - prediction, 2);
    // }
    // std::cout << "Factorised RMSE: " << sqrt(factorised_sum_error / sample_query->RowCount()) << "\n\n";
}

void test_housing(duckdb::Connection &con) {
    // n=1 factorised model ----------------------------------------------------
    std::cout << "\nn=1 factorised model\n";
    // Load datasets 
    con.Query("CREATE TABLE train AS SELECT * FROM read_csv_auto('test/quackml/home_credit/application_train_clean.csv');");
    
    // Training 
    auto start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
    SELECT linear_regression_ring([ring], 0) 
        FROM (
            SELECT to_ring([
                TARGET, CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, OWN_CAR_AGE, FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL, CNT_FAM_MEMBERS, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY, HOUR_APPR_PROCESS_START, REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, APARTMENTS_AVG, BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG, YEARS_BUILD_AVG, COMMONAREA_AVG, ELEVATORS_AVG, ENTRANCES_AVG, FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, LIVINGAPARTMENTS_AVG, LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, APARTMENTS_MODE, BASEMENTAREA_MODE, YEARS_BEGINEXPLUATATION_MODE, YEARS_BUILD_MODE, COMMONAREA_MODE, ELEVATORS_MODE, ENTRANCES_MODE, FLOORSMAX_MODE, FLOORSMIN_MODE, LANDAREA_MODE, LIVINGAPARTMENTS_MODE, LIVINGAREA_MODE, NONLIVINGAPARTMENTS_MODE, NONLIVINGAREA_MODE, APARTMENTS_MEDI, BASEMENTAREA_MEDI, YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_MEDI, COMMONAREA_MEDI, ELEVATORS_MEDI, ENTRANCES_MEDI, FLOORSMAX_MEDI, FLOORSMIN_MEDI, LANDAREA_MEDI, LIVINGAPARTMENTS_MEDI, LIVINGAREA_MEDI, NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_MEDI, TOTALAREA_MODE, OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_2, FLAG_DOCUMENT_3, FLAG_DOCUMENT_4, FLAG_DOCUMENT_5, FLAG_DOCUMENT_6, FLAG_DOCUMENT_7, FLAG_DOCUMENT_8, FLAG_DOCUMENT_9, FLAG_DOCUMENT_10, FLAG_DOCUMENT_11, FLAG_DOCUMENT_12, FLAG_DOCUMENT_13, FLAG_DOCUMENT_14, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_17, FLAG_DOCUMENT_18, FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR
            ]) AS ring 
            FROM 
                train
        ) AS train_ring
    )")->Print();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Train time: " << duration << " mms\n";

    // n=1 unfactorised model --------------------------------------------------
    std::cout << "\nn=1 unfactorised model\n";
    auto train_start_time = std::chrono::high_resolution_clock::now();
    con.Query("SELECT linear_regression([CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, OWN_CAR_AGE, FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL, CNT_FAM_MEMBERS, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY, HOUR_APPR_PROCESS_START, REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, APARTMENTS_AVG, BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG, YEARS_BUILD_AVG, COMMONAREA_AVG, ELEVATORS_AVG, ENTRANCES_AVG, FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, LIVINGAPARTMENTS_AVG, LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, APARTMENTS_MODE, BASEMENTAREA_MODE, YEARS_BEGINEXPLUATATION_MODE, YEARS_BUILD_MODE, COMMONAREA_MODE, ELEVATORS_MODE, ENTRANCES_MODE, FLOORSMAX_MODE, FLOORSMIN_MODE, LANDAREA_MODE, LIVINGAPARTMENTS_MODE, LIVINGAREA_MODE, NONLIVINGAPARTMENTS_MODE, NONLIVINGAREA_MODE, APARTMENTS_MEDI, BASEMENTAREA_MEDI, YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_MEDI, COMMONAREA_MEDI, ELEVATORS_MEDI, ENTRANCES_MEDI, FLOORSMAX_MEDI, FLOORSMIN_MEDI, LANDAREA_MEDI, LIVINGAPARTMENTS_MEDI, LIVINGAREA_MEDI, NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_MEDI, TOTALAREA_MODE, OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_2, FLAG_DOCUMENT_3, FLAG_DOCUMENT_4, FLAG_DOCUMENT_5, FLAG_DOCUMENT_6, FLAG_DOCUMENT_7, FLAG_DOCUMENT_8, FLAG_DOCUMENT_9, FLAG_DOCUMENT_10, FLAG_DOCUMENT_11, FLAG_DOCUMENT_12, FLAG_DOCUMENT_13, FLAG_DOCUMENT_14, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_17, FLAG_DOCUMENT_18, FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR], TARGET, 0) FROM train;")->Print();
    auto train_end_time = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count();
    std::cout << "Train time: " << train_duration << " mms\n";

    // // n=2 factorised model ----------------------------------------------------
    // Load datasets 
    std::cout << "\nn=2 factorised model\n";
    con.Query("CREATE TABLE train AS SELECT * FROM read_csv_auto('test/quackml/home_credit/application_train_clean.csv');");
    con.Query("CREATE TABLE bureau AS SELECT * FROM read_csv_auto('test/quackml/home_credit/bureau_clean.csv');");
    con.Query("CREATE TABLE previous AS SELECT * FROM read_csv_auto('test/quackml/home_credit/previous_application_clean.csv');");

    // Train 
    train_start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        SELECT linear_regression_ring(
            [train_ring.ring, bureau_ring.ring], 0
        ) 
        FROM 
            (SELECT 
                SK_ID_CURR, 
                to_ring([TARGET, CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, OWN_CAR_AGE, FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL, CNT_FAM_MEMBERS, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY, HOUR_APPR_PROCESS_START, REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, APARTMENTS_AVG, BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG, YEARS_BUILD_AVG, COMMONAREA_AVG, ELEVATORS_AVG, ENTRANCES_AVG, FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, LIVINGAPARTMENTS_AVG, LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, APARTMENTS_MODE, BASEMENTAREA_MODE, YEARS_BEGINEXPLUATATION_MODE, YEARS_BUILD_MODE, COMMONAREA_MODE, ELEVATORS_MODE, ENTRANCES_MODE, FLOORSMAX_MODE, FLOORSMIN_MODE, LANDAREA_MODE, LIVINGAPARTMENTS_MODE, LIVINGAREA_MODE, NONLIVINGAPARTMENTS_MODE, NONLIVINGAREA_MODE, APARTMENTS_MEDI, BASEMENTAREA_MEDI, YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_MEDI, COMMONAREA_MEDI, ELEVATORS_MEDI, ENTRANCES_MEDI, FLOORSMAX_MEDI, FLOORSMIN_MEDI, LANDAREA_MEDI, LIVINGAPARTMENTS_MEDI, LIVINGAREA_MEDI, NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_MEDI, TOTALAREA_MODE, OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_2, FLAG_DOCUMENT_3, FLAG_DOCUMENT_4, FLAG_DOCUMENT_5, FLAG_DOCUMENT_6, FLAG_DOCUMENT_7, FLAG_DOCUMENT_8, FLAG_DOCUMENT_9, FLAG_DOCUMENT_10, FLAG_DOCUMENT_11, FLAG_DOCUMENT_12, FLAG_DOCUMENT_13, FLAG_DOCUMENT_14, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_17, FLAG_DOCUMENT_18, FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR]) AS ring 
            FROM 
                train 
            GROUP BY 
                SK_ID_CURR
            ) AS train_ring,
            (SELECT 
                SK_ID_CURR, 
                to_ring([SK_ID_BUREAU, DAYS_CREDIT, CREDIT_DAY_OVERDUE, DAYS_CREDIT_ENDDATE, DAYS_ENDDATE_FACT, AMT_CREDIT_MAX_OVERDUE, CNT_CREDIT_PROLONG, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_LIMIT, AMT_CREDIT_SUM_OVERDUE, DAYS_CREDIT_UPDATE, AMT_ANNUITY]) AS ring 
            FROM 
                bureau 
            GROUP BY 
                SK_ID_CURR
            ) AS bureau_ring
        WHERE 
            train_ring.SK_ID_CURR = bureau_ring.SK_ID_CURR;
    )")->Print();
    train_end_time = std::chrono::high_resolution_clock::now();
    train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count();
    std::cout << "Train time: " << train_duration << " mms\n";

    // n=2 unfactorised model --------------------------------------------------
    std::cout << "\nn=2 unfactorised model\n";
    auto join_start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        CREATE TABLE joined AS 
    SELECT 
        [train.CNT_CHILDREN, train.AMT_INCOME_TOTAL, train.AMT_CREDIT, train.AMT_ANNUITY, train.AMT_GOODS_PRICE, train.REGION_POPULATION_RELATIVE, train.DAYS_BIRTH, train.DAYS_EMPLOYED, train.DAYS_REGISTRATION, train.DAYS_ID_PUBLISH, train.OWN_CAR_AGE, train.FLAG_MOBIL, train.FLAG_EMP_PHONE, train.FLAG_WORK_PHONE, train.FLAG_CONT_MOBILE, train.FLAG_PHONE, train.FLAG_EMAIL, train.CNT_FAM_MEMBERS, train.REGION_RATING_CLIENT, train.REGION_RATING_CLIENT_W_CITY, train.HOUR_APPR_PROCESS_START, train.REG_REGION_NOT_LIVE_REGION, train.REG_REGION_NOT_WORK_REGION, train.LIVE_REGION_NOT_WORK_REGION, train.REG_CITY_NOT_LIVE_CITY, train.REG_CITY_NOT_WORK_CITY, train.LIVE_CITY_NOT_WORK_CITY, train.EXT_SOURCE_1, train.EXT_SOURCE_2, train.EXT_SOURCE_3, train.APARTMENTS_AVG, train.BASEMENTAREA_AVG, train.YEARS_BEGINEXPLUATATION_AVG, train.YEARS_BUILD_AVG, train.COMMONAREA_AVG, train.ELEVATORS_AVG, train.ENTRANCES_AVG, train.FLOORSMAX_AVG, train.FLOORSMIN_AVG, train.LANDAREA_AVG, train.LIVINGAPARTMENTS_AVG, train.LIVINGAREA_AVG, train.NONLIVINGAPARTMENTS_AVG, train.NONLIVINGAREA_AVG, train.APARTMENTS_MODE, train.BASEMENTAREA_MODE, train.YEARS_BEGINEXPLUATATION_MODE, train.YEARS_BUILD_MODE, train.COMMONAREA_MODE, train.ELEVATORS_MODE, train.ENTRANCES_MODE, train.FLOORSMAX_MODE, train.FLOORSMIN_MODE, train.LANDAREA_MODE, train.LIVINGAPARTMENTS_MODE, train.LIVINGAREA_MODE, train.NONLIVINGAPARTMENTS_MODE, train.NONLIVINGAREA_MODE, train.APARTMENTS_MEDI, train.BASEMENTAREA_MEDI, train.YEARS_BEGINEXPLUATATION_MEDI, train.YEARS_BUILD_MEDI, train.COMMONAREA_MEDI, train.ELEVATORS_MEDI, train.ENTRANCES_MEDI, train.FLOORSMAX_MEDI, train.FLOORSMIN_MEDI, train.LANDAREA_MEDI, train.LIVINGAPARTMENTS_MEDI, train.LIVINGAREA_MEDI, train.NONLIVINGAPARTMENTS_MEDI, train.NONLIVINGAREA_MEDI, train.TOTALAREA_MODE, train.OBS_30_CNT_SOCIAL_CIRCLE, train.DEF_30_CNT_SOCIAL_CIRCLE, train.OBS_60_CNT_SOCIAL_CIRCLE, train.DEF_60_CNT_SOCIAL_CIRCLE, train.DAYS_LAST_PHONE_CHANGE, train.FLAG_DOCUMENT_2, train.FLAG_DOCUMENT_3, train.FLAG_DOCUMENT_4, train.FLAG_DOCUMENT_5, train.FLAG_DOCUMENT_6, train.FLAG_DOCUMENT_7, train.FLAG_DOCUMENT_8, train.FLAG_DOCUMENT_9, train.FLAG_DOCUMENT_10, train.FLAG_DOCUMENT_11, train.FLAG_DOCUMENT_12, train.FLAG_DOCUMENT_13, train.FLAG_DOCUMENT_14, train.FLAG_DOCUMENT_15, train.FLAG_DOCUMENT_16, train.FLAG_DOCUMENT_17, train.FLAG_DOCUMENT_18, train.FLAG_DOCUMENT_19, train.FLAG_DOCUMENT_20, train.FLAG_DOCUMENT_21, train.AMT_REQ_CREDIT_BUREAU_HOUR, train.AMT_REQ_CREDIT_BUREAU_DAY, train.AMT_REQ_CREDIT_BUREAU_WEEK, train.AMT_REQ_CREDIT_BUREAU_MON, train.AMT_REQ_CREDIT_BUREAU_QRT, train.AMT_REQ_CREDIT_BUREAU_YEAR, bureau.SK_ID_BUREAU, bureau.DAYS_CREDIT, bureau.CREDIT_DAY_OVERDUE, bureau.DAYS_CREDIT_ENDDATE, bureau.DAYS_ENDDATE_FACT, bureau.AMT_CREDIT_MAX_OVERDUE, bureau.CNT_CREDIT_PROLONG, bureau.AMT_CREDIT_SUM, bureau.AMT_CREDIT_SUM_DEBT, bureau.AMT_CREDIT_SUM_LIMIT, bureau.AMT_CREDIT_SUM_OVERDUE, bureau.DAYS_CREDIT_UPDATE, bureau.AMT_ANNUITY] AS features, 
        train.TARGET
    FROM 
        train 
    JOIN bureau ON train.SK_ID_CURR = bureau.SK_ID_CURR;)");
    auto join_end_time = std::chrono::high_resolution_clock::now();
    auto join_duration = std::chrono::duration_cast<std::chrono::microseconds>(join_end_time - join_start_time).count();
    std::cout << "Join time: " << join_duration << " mms\n";

    train_start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        SELECT linear_regression(features, TARGET, 0)
        FROM joined;)")->Print();
    train_end_time = std::chrono::high_resolution_clock::now();
    train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count();
    std::cout << "Train time: " << train_duration << " mms\n";

    // n=3 factorised model ----------------------------------------------------
    std::cout << "\nn=3 factorised model\n";

    // Train 
    train_start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        SELECT 
            linear_regression_ring([train_ring.ring, bureau_ring.ring, previous_ring.ring], 0) 
        FROM 
            (SELECT 
                SK_ID_CURR, 
                to_ring([TARGET, CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, OWN_CAR_AGE, FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL, CNT_FAM_MEMBERS, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY, HOUR_APPR_PROCESS_START, REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, APARTMENTS_AVG, BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG, YEARS_BUILD_AVG, COMMONAREA_AVG, ELEVATORS_AVG, ENTRANCES_AVG, FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, LIVINGAPARTMENTS_AVG, LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, APARTMENTS_MODE, BASEMENTAREA_MODE, YEARS_BEGINEXPLUATATION_MODE, YEARS_BUILD_MODE, COMMONAREA_MODE, ELEVATORS_MODE, ENTRANCES_MODE, FLOORSMAX_MODE, FLOORSMIN_MODE, LANDAREA_MODE, LIVINGAPARTMENTS_MODE, LIVINGAREA_MODE, NONLIVINGAPARTMENTS_MODE, NONLIVINGAREA_MODE, APARTMENTS_MEDI, BASEMENTAREA_MEDI, YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_MEDI, COMMONAREA_MEDI, ELEVATORS_MEDI, ENTRANCES_MEDI, FLOORSMAX_MEDI, FLOORSMIN_MEDI, LANDAREA_MEDI, LIVINGAPARTMENTS_MEDI, LIVINGAREA_MEDI, NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_MEDI, TOTALAREA_MODE, OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_2, FLAG_DOCUMENT_3, FLAG_DOCUMENT_4, FLAG_DOCUMENT_5, FLAG_DOCUMENT_6, FLAG_DOCUMENT_7, FLAG_DOCUMENT_8, FLAG_DOCUMENT_9, FLAG_DOCUMENT_10, FLAG_DOCUMENT_11, FLAG_DOCUMENT_12, FLAG_DOCUMENT_13, FLAG_DOCUMENT_14, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_17, FLAG_DOCUMENT_18, FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR]) AS ring 
            FROM 
                train 
            GROUP BY 
                SK_ID_CURR
            ) AS train_ring
        JOIN 
            (SELECT 
                SK_ID_CURR, 
                to_ring([SK_ID_BUREAU, DAYS_CREDIT, CREDIT_DAY_OVERDUE, DAYS_CREDIT_ENDDATE, DAYS_ENDDATE_FACT, AMT_CREDIT_MAX_OVERDUE, CNT_CREDIT_PROLONG, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_LIMIT, AMT_CREDIT_SUM_OVERDUE, DAYS_CREDIT_UPDATE, AMT_ANNUITY]) AS ring 
            FROM 
                bureau 
            GROUP BY 
                SK_ID_CURR
            ) AS bureau_ring
        ON 
            train_ring.SK_ID_CURR = bureau_ring.SK_ID_CURR
        JOIN 
            (SELECT 
                SK_ID_CURR, 
                to_ring([SK_ID_PREV, SK_ID_CURR, AMT_ANNUITY, AMT_APPLICATION, AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, HOUR_APPR_PROCESS_START, NFLAG_LAST_APPL_IN_DAY, RATE_DOWN_PAYMENT, RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED, DAYS_DECISION, SELLERPLACE_AREA, CNT_PAYMENT, DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE, DAYS_TERMINATION, NFLAG_INSURED_ON_APPROVAL]) AS ring 
            FROM 
                previous 
            GROUP BY 
                SK_ID_CURR
            ) AS previous_ring
        ON 
            train_ring.SK_ID_CURR = previous_ring.SK_ID_CURR;
    )")->Print();
    train_end_time = std::chrono::high_resolution_clock::now();
    train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count();
    std::cout << "Train time: " << train_duration << " mms\n";
    

    // n=3 unfactorised model --------------------------------------------------
    std::cout << "\nn=3 unfactorised model\n";
    con.Query("CREATE TABLE previous AS SELECT * FROM read_csv_auto('test/quackml/home_credit/previous_application_clean.csv');");
    join_start_time = std::chrono::high_resolution_clock::now();
    train_start_time = std::chrono::high_resolution_clock::now();
    con.Query(R"(
        SELECT linear_regression(
            [train.CNT_CHILDREN, train.AMT_INCOME_TOTAL, train.AMT_CREDIT, train.AMT_ANNUITY, train.AMT_GOODS_PRICE, train.REGION_POPULATION_RELATIVE, train.DAYS_BIRTH, train.DAYS_EMPLOYED, train.DAYS_REGISTRATION, train.DAYS_ID_PUBLISH, train.OWN_CAR_AGE, train.FLAG_MOBIL, train.FLAG_EMP_PHONE, train.FLAG_WORK_PHONE, train.FLAG_CONT_MOBILE, train.FLAG_PHONE, train.FLAG_EMAIL, train.CNT_FAM_MEMBERS, train.REGION_RATING_CLIENT, train.REGION_RATING_CLIENT_W_CITY, train.HOUR_APPR_PROCESS_START, train.REG_REGION_NOT_LIVE_REGION, train.REG_REGION_NOT_WORK_REGION, train.LIVE_REGION_NOT_WORK_REGION, train.REG_CITY_NOT_LIVE_CITY, train.REG_CITY_NOT_WORK_CITY, train.LIVE_CITY_NOT_WORK_CITY, train.EXT_SOURCE_1, train.EXT_SOURCE_2, train.EXT_SOURCE_3, train.APARTMENTS_AVG, train.BASEMENTAREA_AVG, train.YEARS_BEGINEXPLUATATION_AVG, train.YEARS_BUILD_AVG, train.COMMONAREA_AVG, train.ELEVATORS_AVG, train.ENTRANCES_AVG, train.FLOORSMAX_AVG, train.FLOORSMIN_AVG, train.LANDAREA_AVG, train.LIVINGAPARTMENTS_AVG, train.LIVINGAREA_AVG, train.NONLIVINGAPARTMENTS_AVG, train.NONLIVINGAREA_AVG, train.APARTMENTS_MODE, train.BASEMENTAREA_MODE, train.YEARS_BEGINEXPLUATATION_MODE, train.YEARS_BUILD_MODE, train.COMMONAREA_MODE, train.ELEVATORS_MODE, train.ENTRANCES_MODE, train.FLOORSMAX_MODE, train.FLOORSMIN_MODE, train.LANDAREA_MODE, train.LIVINGAPARTMENTS_MODE, train.LIVINGAREA_MODE, train.NONLIVINGAPARTMENTS_MODE, train.NONLIVINGAREA_MODE, train.APARTMENTS_MEDI, train.BASEMENTAREA_MEDI, train.YEARS_BEGINEXPLUATATION_MEDI, train.YEARS_BUILD_MEDI, train.COMMONAREA_MEDI, train.ELEVATORS_MEDI, train.ENTRANCES_MEDI, train.FLOORSMAX_MEDI, train.FLOORSMIN_MEDI, train.LANDAREA_MEDI, train.LIVINGAPARTMENTS_MEDI, train.LIVINGAREA_MEDI, train.NONLIVINGAPARTMENTS_MEDI, train.NONLIVINGAREA_MEDI, train.TOTALAREA_MODE, train.OBS_30_CNT_SOCIAL_CIRCLE, train.DEF_30_CNT_SOCIAL_CIRCLE, train.OBS_60_CNT_SOCIAL_CIRCLE, train.DEF_60_CNT_SOCIAL_CIRCLE, train.DAYS_LAST_PHONE_CHANGE, train.FLAG_DOCUMENT_2, train.FLAG_DOCUMENT_3, train.FLAG_DOCUMENT_4, train.FLAG_DOCUMENT_5, train.FLAG_DOCUMENT_6, train.FLAG_DOCUMENT_7, train.FLAG_DOCUMENT_8, train.FLAG_DOCUMENT_9, train.FLAG_DOCUMENT_10, train.FLAG_DOCUMENT_11, train.FLAG_DOCUMENT_12, train.FLAG_DOCUMENT_13, train.FLAG_DOCUMENT_14, train.FLAG_DOCUMENT_15, train.FLAG_DOCUMENT_16, train.FLAG_DOCUMENT_17, train.FLAG_DOCUMENT_18, train.FLAG_DOCUMENT_19, train.FLAG_DOCUMENT_20, train.FLAG_DOCUMENT_21, train.AMT_REQ_CREDIT_BUREAU_HOUR, train.AMT_REQ_CREDIT_BUREAU_DAY, train.AMT_REQ_CREDIT_BUREAU_WEEK, train.AMT_REQ_CREDIT_BUREAU_MON, train.AMT_REQ_CREDIT_BUREAU_QRT, train.AMT_REQ_CREDIT_BUREAU_YEAR, bureau.SK_ID_BUREAU, bureau.DAYS_CREDIT, bureau.CREDIT_DAY_OVERDUE, bureau.DAYS_CREDIT_ENDDATE, bureau.DAYS_ENDDATE_FACT, bureau.AMT_CREDIT_MAX_OVERDUE, bureau.CNT_CREDIT_PROLONG, bureau.AMT_CREDIT_SUM, bureau.AMT_CREDIT_SUM_DEBT, bureau.AMT_CREDIT_SUM_LIMIT, bureau.AMT_CREDIT_SUM_OVERDUE, bureau.DAYS_CREDIT_UPDATE, bureau.AMT_ANNUITY , previous.SK_ID_PREV , previous.SK_ID_CURR , previous.AMT_ANNUITY , previous.AMT_APPLICATION , previous.AMT_CREDIT , previous.AMT_DOWN_PAYMENT , previous.AMT_GOODS_PRICE , previous.HOUR_APPR_PROCESS_START , previous.NFLAG_LAST_APPL_IN_DAY , previous.RATE_DOWN_PAYMENT , previous.RATE_INTEREST_PRIMARY , previous.RATE_INTEREST_PRIVILEGED , previous.DAYS_DECISION , previous.SELLERPLACE_AREA , previous.CNT_PAYMENT , previous.DAYS_FIRST_DRAWING , previous.DAYS_FIRST_DUE , previous.DAYS_LAST_DUE_1ST_VERSION , previous.DAYS_LAST_DUE , previous.DAYS_TERMINATION , previous.NFLAG_INSURED_ON_APPROVAL],
            train.TARGET,
            0)
        FROM
            train
        JOIN bureau ON train.SK_ID_CURR = bureau.SK_ID_CURR
        JOIN previous ON train.SK_ID_CURR = previous.SK_ID_CURR;
    )")->Print();
    train_end_time = std::chrono::high_resolution_clock::now();
    train_duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end_time - train_start_time).count();
    std::cout << "Train time: " << train_duration << " mms\n";

}

void run_quackml_tests(DuckDB &db) {
    std::cout << "<=========== Running QuackML tests ===========>\n";
    Connection con(db);

    // Test regular linear regression 
    test_regression(con);

    // Housing dataset 
    test_housing(con);

    // nycflights13 dataset
    test_flights(con);

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

    // run_quackml_tests(db);
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
