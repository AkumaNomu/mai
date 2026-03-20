from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Sequence

import pandas as pd


DEFAULT_TABLE_NAME = 'cache_rows'


def resolve_sqlite_cache_path(path: str | None, *, default_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(str(path or default_path))
    lowered = normalized.lower()
    if lowered.endswith('.sqlite') or lowered.endswith('.db'):
        return normalized, f'{os.path.splitext(normalized)[0]}.csv'
    if lowered.endswith('.csv'):
        return f'{os.path.splitext(normalized)[0]}.sqlite', normalized
    return f'{normalized}.sqlite', f'{normalized}.csv'


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _ordered_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    ordered = [column for column in columns if column in df.columns]
    ordered.extend(sorted(column for column in df.columns if column not in ordered))
    return ordered


def read_sqlite_table(
    path: str,
    *,
    columns: Sequence[str],
    table_name: str = DEFAULT_TABLE_NAME,
) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(columns))
    connection = sqlite3.connect(path)
    try:
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (str(table_name),),
        )
        if cursor.fetchone() is None:
            return pd.DataFrame(columns=list(columns))
        df = pd.read_sql_query(
            f'SELECT * FROM {_quote_identifier(table_name)}',
            connection,
        )
    finally:
        connection.close()
    if df.empty:
        return pd.DataFrame(columns=list(columns))
    return df.reindex(columns=_ordered_columns(df, columns))


def write_sqlite_table(
    path: str,
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    table_name: str = DEFAULT_TABLE_NAME,
    key_columns: Sequence[str] = (),
) -> None:
    if not path:
        return
    ordered_columns = _ordered_columns(df, columns)
    prepared = df.reindex(columns=ordered_columns) if ordered_columns else df.copy()
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='mai_cache_', suffix='.sqlite.tmp', dir=directory)
    os.close(fd)
    try:
        connection = sqlite3.connect(temp_path)
        try:
            prepared.to_sql(str(table_name), connection, index=False, if_exists='replace')
            if key_columns:
                index_name = f'{str(table_name)}_pk_idx'
                quoted_keys = ', '.join(_quote_identifier(column) for column in key_columns)
                connection.execute(
                    f'CREATE UNIQUE INDEX {_quote_identifier(index_name)} '
                    f'ON {_quote_identifier(table_name)} ({quoted_keys})'
                )
            connection.commit()
        finally:
            connection.close()
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
