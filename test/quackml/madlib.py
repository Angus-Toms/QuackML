import psycopg2
from psycopg2 import Error

try:
    # Establish a connection to the database
    conn = psycopg2.connect(
        dbname="postgres",
        host="localhost",
        user="postgres",
        password='08082002',
        port="5433"
    )

    # Create a cursor object
    cur = conn.cursor()

    # Execute a SQL command
    cur.execute("CREATE EXTENSION madlib;")
    cur.execute("SELECT * FROM tbl_1;")

    # Fetch the result
    rows = cur.fetchall()
    for row in rows:
        print(row)

except Error as e:
    print("Error connecting to PostgreSQL database:", e)

finally:
    # Close cursor and connection
    if conn is not None:
        conn.commit()
        cur.close()
        conn.close()