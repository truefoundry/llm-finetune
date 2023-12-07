import json
import logging
from urllib.parse import parse_qsl, urlparse

logger = logging.getLogger("finetune")
# URI format:
# snowflake://{user}:{password}@{account}/{database}/{schema}/{table}?warehouse={warehouse}&role={role}
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"


def get_data_from_snowflake_table(
    uri: str,
    max_num_samples: int = 0,
    batch_size=500,
):
    import snowflake.connector

    parsed_uri = urlparse(uri)
    database, schema, table, *_ = parsed_uri.path.strip("/").split("/")
    query_params = dict(parse_qsl(parsed_uri.query))
    kwargs = dict(
        user=parsed_uri.username,
        password="*******",
        account=parsed_uri.hostname,
        database=database,
        schema=schema,
        **query_params,
    )
    logger.info(f"Connecting to Snowflake table {table} with args: {kwargs}")
    kwargs["password"] = parsed_uri.password
    connection, cursor = None, None
    try:
        connection = snowflake.connector.connect(**kwargs)
        cursor = connection.cursor()
        if max_num_samples > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT {max_num_samples}")
        else:
            cursor.execute(f"SELECT * FROM {table}")
        column_names = [column_name.name for column_name in cursor.description]
        print(f"Got columns: {column_names}")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                line = dict(zip(column_names, row))
                line = {
                    PROMPT_KEY: line.get(PROMPT_KEY) or line.get("PROMPT"),
                    COMPLETION_KEY: line.get(COMPLETION_KEY) or line.get("COMPLETION"),
                }
                yield json.dumps(line)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
