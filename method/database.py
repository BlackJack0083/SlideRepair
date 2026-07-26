import os
from functools import lru_cache
from urllib.parse import quote_plus

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def _database_engine() -> Engine:
    """Create the database engine from environment variables on first use."""
    database_url = (
        "postgresql+psycopg2://"
        f"{quote_plus(os.environ['SQL_USER'])}:{quote_plus(os.environ['SQL_PASSWORD'])}"
        f"@{os.environ['SQL_HOST']}:{os.getenv('SQL_PORT', '5432')}"
        f"/{os.environ['SQL_DB']}"
    )
    logger.info("Method database engine initialized for host: {}", os.environ["SQL_HOST"])
    return create_engine(database_url)


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """执行 SQL 查询。

    Args:
        sql: SQL 文本，动态值必须通过 `params` 绑定。
        params: SQLAlchemy 参数字典。

    Returns:
        查询结果 dataframe。
    """
    try:
        with _database_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:
        logger.error(f"Method query failed: {exc}\nSQL: {sql}")
        raise
