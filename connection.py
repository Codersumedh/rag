"""Snowflake connection helper.

Uses the same connect method you shared (externalbrowser auth).
A browser window will open the first time to log you in.
"""

import pandas as pd
import snowflake.connector as snow

from config import SNOWFLAKE_CONN


def get_connection():
    """Open and return a Snowflake connection."""
    conn = snow.connect(**SNOWFLAKE_CONN)
    print("Connected to Snowflake")
    return conn


def run_query(sql: str, conn=None) -> pd.DataFrame:
    """Run a SQL string and return the result as a pandas DataFrame.

    If no connection is passed, a new one is opened and closed for you.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        if own_conn:
            conn.close()
