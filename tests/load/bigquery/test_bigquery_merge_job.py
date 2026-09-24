from typing import Any
from unittest.mock import Mock

import pytest

from dlt.common.destination import PreparedTableSchema
from dlt.destinations.exceptions import DestinationSchemaWillNotUpdate
from dlt.destinations.impl.bigquery.bigquery import BigQueryMergeJob
from dlt.destinations.impl.bigquery.bigquery_adapter import PARTITION_HINT
from dlt.destinations.impl.bigquery import sql_client as bigquery_sql_client
from dlt.destinations.sql_jobs import SqlMergeFollowupJob


class _SqlClient:
    class capabilities:
        escape_literal = staticmethod(lambda value: f"'{value}'")

    def escape_column_name(self, column_name: str) -> str:
        return f"`{column_name}`"

    def get_qualified_table_names(self, table_name: str) -> tuple[str, str]:
        return (f"`dataset.{table_name}`", f"`staging.{table_name}`")


def _table(data_type: str, **column_properties: Any) -> PreparedTableSchema:
    return {
        "name": "events",
        "columns": {
            "event_id": {"name": "event_id", "data_type": "bigint", "primary_key": True},
            "event_time": {
                "name": "event_time",
                "data_type": data_type,
                **column_properties,
            },
        },
    }


def test_partition_clause_is_empty_without_partition_hint() -> None:
    clause = BigQueryMergeJob.gen_partition_clause(_table("date"), _SqlClient())  # type: ignore[arg-type]

    assert clause == ""


def test_partition_clause_filters_date_partitions_with_constant_literals() -> None:
    clause = BigQueryMergeJob.gen_partition_clause(
        _table("date", partition=True, partition_values=["2026-09-03", "2026-09-02"]),
        _SqlClient(),  # type: ignore[arg-type]
    )

    assert clause == " AND d.`event_time` IN (DATE '2026-09-02', DATE '2026-09-03')"


def test_partition_clause_converts_timestamp_to_date_literals() -> None:
    clause = BigQueryMergeJob.gen_partition_clause(
        _table(
            "timestamp",
            **{PARTITION_HINT: True, "partition_values": ["2026-09-02"]},
        ),
        _SqlClient(),  # type: ignore[arg-type]
    )

    assert clause == " AND DATE(d.`event_time`) IN (DATE '2026-09-02')"


def test_upsert_merge_includes_partition_predicate() -> None:
    sql = BigQueryMergeJob.gen_upsert_sql(
        [_table("date", partition=True, partition_values=["2026-09-02"])],
        _SqlClient(),  # type: ignore[arg-type]
    )

    assert "ON d.`event_id` = s.`event_id` AND d.`event_time` IN (DATE '2026-09-02')" in sql[0]


def test_partition_clause_is_empty_without_partition_values() -> None:
    clause = BigQueryMergeJob.gen_partition_clause(
        _table("date", partition=True), _SqlClient()  # type: ignore[arg-type]
    )

    assert clause == ""


def test_partition_clause_is_empty_for_unsupported_partition_type() -> None:
    clause = BigQueryMergeJob.gen_partition_clause(
        _table("bigint", **{PARTITION_HINT: True}), _SqlClient()  # type: ignore[arg-type]
    )

    assert clause == ""


def test_partition_clause_rejects_multiple_partition_columns() -> None:
    table = _table("date", **{PARTITION_HINT: True})
    table["columns"]["event_id"][PARTITION_HINT] = True

    with pytest.raises(DestinationSchemaWillNotUpdate):
        BigQueryMergeJob.gen_partition_clause(table, _SqlClient())  # type: ignore[arg-type]


def test_default_sql_merge_job_does_not_add_partition_clause() -> None:
    clause = SqlMergeFollowupJob.gen_partition_clause(_table("date"), _SqlClient())  # type: ignore[arg-type]

    assert clause == ""


def test_script_child_job_id_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Mock()
    cursor = Mock(description=None)
    cursor.query_job = Mock(job_id="script_job", statement_type=None)
    connection.cursor.return_value = cursor

    client = bigquery_sql_client.BigQuerySqlClient.__new__(bigquery_sql_client.BigQuerySqlClient)
    client._client = Mock()
    client._client.list_jobs.return_value = [Mock(job_id="merge_job")]
    client._session_query = None
    client._default_query = Mock()
    monkeypatch.setattr(bigquery_sql_client, "DbApiConnection", Mock(return_value=connection))
    info = Mock()
    monkeypatch.setattr(bigquery_sql_client.logger, "info", info)

    with client.execute_query("CREATE TABLE x; MERGE INTO x USING y ON x.id = y.id"):
        pass

    info.assert_any_call(
        "Submitted BigQuery script child job %s (parent %s)", "merge_job", "script_job"
    )
