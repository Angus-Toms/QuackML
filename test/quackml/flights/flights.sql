-- Import from CSV -- infer schema 
CREATE TABLE airlines AS FROM read_csv_auto('test/quackml/flights/airlines.csv', header=TRUE);
CREATE TABLE airports AS FROM read_csv_auto('test/quackml/flights/airports.csv', header=TRUE);
CREATE TABLE flights AS FROM read_csv_auto('test/quackml/flights/flights.csv', header=TRUE);
CREATE TABLE planes AS FROM read_csv_auto('test/quackml/flights/planes.csv', header=TRUE);
CREATE TABLE weather AS FROM read_csv_auto('test/quackml/flights/weather.csv', header=TRUE);

/* Task: Predict flights.arr_delay as linear function of all other columns */ 

/* Drop flights.dep_time and flights.arr_time - use flights.hour and flights.minute instead */
ALTER TABLE flights DROP COLUMN dep_time;
ALTER TABLE flights DROP COLUMN arr_time;
/* Drop planes.speed - too many NAs */
ALTER TABLE planes DROP COLUMN speed;
/* Drop weather.wind_gust - too many NAs */
ALTER TABLE weather DROP COLUMN wind_gust;
/* Drop weather.year,month,day,hour - use time_hour instead */
ALTER TABLE weather DROP COLUMN year;
ALTER TABLE weather DROP COLUMN month;
ALTER TABLE weather DROP COLUMN day;
ALTER TABLE weather DROP COLUMN hour;

/* Convert flights.dep_delay to int64 */
ALTER TABLE flights ADD COLUMN dep_delay_int INTEGER;
UPDATE flights SET dep_delay_int = TRY_CAST(dep_delay AS INTEGER); 
ALTER TABLE flights DROP COLUMN dep_delay;
DELETE FROM flights WHERE dep_delay_int IS NULL;

/* Convert flights.arr_delay to int64 */
ALTER TABLE flights ADD COLUMN arr_delay_int INT64;
UPDATE flights SET arr_delay_int = TRY_CAST(arr_delay AS INT64); 
ALTER TABLE flights DROP COLUMN arr_delay;
DELETE FROM flights WHERE arr_delay_int IS NULL;

/* Convert flight.air_time to int64 */ 
ALTER TABLE flights ADD COLUMN air_time_int INT64;
UPDATE flights SET air_time_int = TRY_CAST(air_time AS INT64);
ALTER TABLE flights DROP COLUMN air_time;
DELETE FROM flights WHERE air_time_int IS NULL;

/* Convert planes.year to int64 */ 
ALTER TABLE planes ADD COLUMN year_int INT64;
UPDATE planes SET year_int = TRY_CAST(year AS INT64);
ALTER TABLE planes DROP COLUMN year;
DELETE FROM planes WHERE year_int IS NULL;

/* Convert weather.temp to double */
ALTER TABLE weather ADD COLUMN temp_double DOUBLE;
UPDATE weather SET temp_double = TRY_CAST(temp AS DOUBLE);
ALTER TABLE weather DROP COLUMN temp;
DELETE FROM weather WHERE temp_double IS NULL;

/* Convert weather.dewp to double */
ALTER TABLE weather ADD COLUMN dewp_double DOUBLE;
UPDATE weather SET dewp_double = TRY_CAST(dewp AS DOUBLE);
ALTER TABLE weather DROP COLUMN dewp;
DELETE FROM weather WHERE dewp_double IS NULL;

/* Convert weather.humid to double */
ALTER TABLE weather ADD COLUMN humid_double DOUBLE;
UPDATE weather SET humid_double = TRY_CAST(humid AS DOUBLE);
ALTER TABLE weather DROP COLUMN humid;
DELETE FROM weather WHERE humid_double IS NULL;

/* Convert weather.wind_speed to int */
ALTER TABLE weather ADD COLUMN wind_speed_int INTEGER;
UPDATE weather SET wind_speed_int = TRY_CAST(wind_speed AS INTEGER);
ALTER TABLE weather DROP COLUMN wind_speed;
DELETE FROM weather WHERE wind_speed_int IS NULL;

/* Convert weather.pressure to double */
ALTER TABLE weather ADD COLUMN pressure_double DOUBLE;
UPDATE weather SET pressure_double = TRY_CAST(pressure AS DOUBLE);
ALTER TABLE weather DROP COLUMN pressure;
DELETE FROM weather WHERE pressure_double IS NULL;


/* 
================================================================================
One hot encode airports.tzone
================================================================================
*/

-- Define macro
CREATE MACRO one_hot_encode(column_name, category) AS (
    CASE WHEN column_name = category THEN 1 ELSE 0 END
);

/* Encode airports.tzone */
ALTER TABLE airports
ADD COLUMN Asia_Chongqing TINYINT;
ALTER TABLE airports
ADD COLUMN Pacific_Honolulu TINYINT;
ALTER TABLE airports
ADD COLUMN America_Chicago TINYINT;
ALTER TABLE airports
ADD COLUMN NA TINYINT;
ALTER TABLE airports
ADD COLUMN America_Los_Angeles TINYINT;
ALTER TABLE airports
ADD COLUMN America_Vancouver TINYINT;
ALTER TABLE airports
ADD COLUMN America_Anchorage TINYINT;
ALTER TABLE airports
ADD COLUMN America_Denver TINYINT;
ALTER TABLE airports
ADD COLUMN America_New_York TINYINT;
ALTER TABLE airports
ADD COLUMN America_Phoenix TINYINT;
UPDATE airports 
SET 
    Asia_Chongqing = one_hot_encode(tzone, 'Asia/Chongqing'),
    Pacific_Honolulu = one_hot_encode(tzone, 'Pacific/Honolulu'),
    America_Chicago = one_hot_encode(tzone, 'America/Chicago'),
    NA = one_hot_encode(tzone, 'NA'),
    America_Los_Angeles = one_hot_encode(tzone, 'America/Los_Angeles'),
    America_Vancouver = one_hot_encode(tzone, 'America/Vancouver'),
    America_Anchorage = one_hot_encode(tzone, 'America/Anchorage'),
    America_Denver = one_hot_encode(tzone, 'America/Denver'),
    America_New_York = one_hot_encode(tzone, 'America/New_York'),
    America_Phoenix = one_hot_encode(tzone, 'America/Phoenix');
ALTER TABLE airports DROP COLUMN tzone;


/* 
================================================================================
Mean encode flights.dest
================================================================================
*/
-- Convert flights.arr_delay to integer, convert 'NA' to NULL */

CREATE VIEW airport_dest_mean_delay AS
SELECT
    airports.faa AS faa,
    AVG(flights.arr_delay_int) AS mean_arr_delay
FROM
    flights
JOIN
    airports ON flights.dest = airports.faa
GROUP BY
    airports.faa;

ALTER TABLE flights
ADD COLUMN dest_mean_delay DOUBLE;
UPDATE flights 
SET dest_mean_delay = (SELECT mean_arr_delay FROM airport_dest_mean_delay WHERE faa = flights.dest);
UPDATE flights 
SET dest_mean_delay = 0 WHERE dest_mean_delay IS NULL;
ALTER TABLE flights DROP COLUMN dest;

/* 
================================================================================
Mean encode flights.origin
================================================================================
*/
CREATE VIEW airport_origin_mean_delay AS 
SELECT 
    airports.faa AS faa,
    AVG(flights.arr_delay_int) AS mean_arr_delay
FROM 
    flights
JOIN 
    airports ON flights.origin = airports.faa
GROUP BY 
    airports.faa;

ALTER TABLE flights
ADD COLUMN origin_mean_delay DOUBLE;
UPDATE flights 
SET origin_mean_delay = (SELECT mean_arr_delay FROM airport_origin_mean_delay WHERE faa = flights.origin);


/* 
================================================================================
Mean encode planes.model
================================================================================
*/
CREATE VIEW plane_model_mean_delay AS 
SELECT 
    planes.model AS model,
    AVG(flights.arr_delay_int) AS mean_arr_delay
FROM 
    flights 
JOIN 
    planes ON flights.tailnum = planes.tailnum
GROUP BY
    planes.model;

ALTER TABLE planes 
ADD COLUMN model_mean_delay DOUBLE;
UPDATE planes
SET model_mean_delay = (SELECT mean_arr_delay FROM plane_model_mean_delay WHERE model = planes.model);
ALTER TABLE planes DROP COLUMN model;


/* 
================================================================================
Mean encode planes.manufacturer
================================================================================
*/
CREATE VIEW plane_manufacturer_mean_delay AS
SELECT 
    planes.manufacturer AS manufacturer,
    AVG(flights.arr_delay_int) AS mean_arr_delay
FROM 
    flights
JOIN 
    planes ON flights.tailnum = planes.tailnum
GROUP BY
    planes.manufacturer;

ALTER TABLE planes
ADD COLUMN manufacturer_mean_delay DOUBLE;
UPDATE planes
SET manufacturer_mean_delay = (SELECT mean_arr_delay FROM plane_manufacturer_mean_delay WHERE manufacturer = planes.manufacturer);
ALTER TABLE planes DROP COLUMN manufacturer;


/* 
================================================================================
One-hot encode planes.type
================================================================================
*/
ALTER TABLE planes 
ADD COLUMN fixed_wing_multi TINYINT;
ALTER TABLE planes
ADD COLUMN fixed_wing_single TINYINT;
ALTER TABLE planes
ADD COLUMN rotorcraft TINYINT;
UPDATE planes 
SET 
    fixed_wing_multi = one_hot_encode(type, 'Fixed wing multi engine'),
    fixed_wing_single = one_hot_encode(type, 'Fixed wing single engine'),
    rotorcraft = one_hot_encode(type, 'Rotorcraft');
ALTER TABLE planes DROP COLUMN type;

/* 
================================================================================
One-hot encode planes.engine
================================================================================
*/
ALTER TABLE planes 
ADD COLUMN turbo_jet TINYINT;
ALTER TABLE planes
ADD COLUMN turbo_prop TINYINT;
ALTER TABLE planes
ADD COLUMN turbo_shaft TINYINT;
ALTER TABLE planes
ADD COLUMN turbo_fan TINYINT;
ALTER TABLE planes
ADD COLUMN reciprocating TINYINT;
ALTER TABLE planes
ADD COLUMN four_cycle TINYINT;
UPDATE planes
SET 
    turbo_jet = one_hot_encode(engine, 'Turbo-jet'),
    turbo_prop = one_hot_encode(engine, 'Turbo-prop'),
    turbo_shaft = one_hot_encode(engine, 'Turbo-shaft'),
    turbo_fan = one_hot_encode(engine, 'Turbo-fan'),
    reciprocating = one_hot_encode(engine, 'Reciprocating'),
    four_cycle = one_hot_encode(engine, '4 Cycle');
ALTER TABLE planes DROP COLUMN engine;


/*
================================================================================
Mean encode airlines.name 
================================================================================
*/
CREATE VIEW airline_name_mean_delay AS
SELECT 
    airlines.name AS name,
    AVG(flights.arr_delay_int) AS mean_arr_delay
FROM
    flights
JOIN    
    airlines ON flights.carrier = airlines.carrier
GROUP BY
    airlines.name;

ALTER TABLE airlines
ADD COLUMN name_mean_delay DOUBLE;
UPDATE airlines
SET name_mean_delay = (SELECT mean_arr_delay FROM airline_name_mean_delay WHERE name = airlines.name);

SELECT * FROM flights
JOIN airlines ON flights.carrier = airlines.carrier
JOIN airports ON flights.origin = airports.faa
JOIN planes ON flights.tailnum = planes.tailnum
JOIN weather ON flights.origin = weather.origin AND flights.time_hour = weather.time_hour;

COPY airlines TO 'test/quackml/flights/airlines_clean.csv' (HEADER, DELIMITER ',');
COPY airports TO 'test/quackml/flights/airports_clean.csv' (HEADER, DELIMITER ',');
COPY flights TO 'test/quackml/flights/flights_clean.csv' (HEADER, DELIMITER ',');
COPY planes TO 'test/quackml/flights/planes_clean.csv' (HEADER, DELIMITER ',');
COPY weather TO 'test/quackml/flights/weather_clean.csv' (HEADER, DELIMITER ',');

SELECT linear_regression(
    [f.]
)