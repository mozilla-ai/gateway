"""Integration tests for the bulk usage endpoint."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import UsageLog, User

USAGE_PATH = "/v1/usage"


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _make_log(
    db: Session,
    *,
    user_id: str,
    timestamp: datetime,
    api_key_id: str | None = None,
    model: str = "gpt-4",
    provider: str | None = "openai",
    endpoint: str = "/v1/chat/completions",
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    total_tokens: int | None = 15,
    cost: float | None = 0.01,
    status: str = "success",
    error_message: str | None = None,
    log_id: str | None = None,
) -> UsageLog:
    _ensure_user(db, user_id)
    log = UsageLog(
        id=log_id or str(uuid.uuid4()),
        user_id=user_id,
        api_key_id=api_key_id,
        timestamp=timestamp,
        model=model,
        provider=provider,
        endpoint=endpoint,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        status=status,
        error_message=error_message,
    )
    db.add(log)
    return log


def test_list_usage_requires_master_key(client: TestClient) -> None:
    response = client.get(USAGE_PATH)
    assert response.status_code == 401


def test_list_usage_returns_empty_when_no_logs(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    assert response.json() == []


def test_list_usage_orders_newest_first(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    older = now - timedelta(hours=1)
    _make_log(db_session, user_id="order-user", timestamp=older)
    _make_log(db_session, user_id="order-user", timestamp=now)
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert [entry["timestamp"] for entry in data] == [now.isoformat(), older.isoformat()]


def test_list_usage_filter_by_start_date(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    earlier = datetime(2025, 2, 1, 10, 0, tzinfo=UTC)
    later = datetime(2025, 2, 1, 12, 0, tzinfo=UTC)
    _make_log(db_session, user_id="start-user", timestamp=earlier)
    _make_log(db_session, user_id="start-user", timestamp=later)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"start_date": later.isoformat()},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["timestamp"] == later.isoformat()


def test_list_usage_filter_by_end_date(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    earlier = datetime(2025, 3, 1, 8, 0, tzinfo=UTC)
    later = datetime(2025, 3, 1, 9, 0, tzinfo=UTC)
    _make_log(db_session, user_id="end-user", timestamp=earlier)
    _make_log(db_session, user_id="end-user", timestamp=later)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"end_date": later.isoformat()},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["timestamp"] == earlier.isoformat()


def test_list_usage_filter_by_time_range(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    base = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    before = base - timedelta(hours=1)
    inside = base + timedelta(minutes=30)
    after = base + timedelta(hours=2)
    _make_log(db_session, user_id="range-user", timestamp=before)
    _make_log(db_session, user_id="range-user", timestamp=base)
    _make_log(db_session, user_id="range-user", timestamp=inside)
    _make_log(db_session, user_id="range-user", timestamp=after)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={
            "start_date": base.isoformat(),
            "end_date": (base + timedelta(hours=2)).isoformat(),
        },
    )
    assert response.status_code == 200
    data = response.json()
    timestamps = [entry["timestamp"] for entry in data]
    assert timestamps == [inside.isoformat(), base.isoformat()]


def test_list_usage_filter_by_user_id(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 5, 1, 15, 0, tzinfo=UTC)
    _make_log(db_session, user_id="filter-a", timestamp=timestamp)
    _make_log(db_session, user_id="filter-b", timestamp=timestamp)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"user_id": "filter-b"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["user_id"] == "filter-b"


def test_list_usage_pagination(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    base = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
    timestamps = []
    for idx in range(5):
        ts = base + timedelta(minutes=idx)
        timestamps.append(ts)
        _make_log(db_session, user_id="pager", timestamp=ts)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"skip": 1, "limit": 2},
    )
    assert response.status_code == 200
    data = response.json()
    expected_order = [ts.isoformat() for ts in sorted(timestamps, reverse=True)]
    assert [entry["timestamp"] for entry in data] == expected_order[1:3]


def test_list_usage_response_shape(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    timestamp = datetime(2025, 7, 1, 9, 30, tzinfo=UTC)
    log = _make_log(
        db_session,
        user_id="shape-user",
        timestamp=timestamp,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/custom",
        prompt_tokens=42,
        completion_tokens=7,
        total_tokens=49,
        cost=1.23,
        status="error",
        error_message="capacity",
        log_id="shape-log-id",
    )
    db_session.commit()

    response = client.get(USAGE_PATH, headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0] == {
        "id": log.id,
        "user_id": "shape-user",
        "api_key_id": None,
        "timestamp": timestamp.isoformat(),
        "model": "gpt-4o",
        "provider": "openai",
        "endpoint": "/custom",
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
        "cost": 1.23,
        "status": "error",
        "error_message": "capacity",
    }


def test_list_usage_limit_max_enforced(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"limit": 1001},
    )
    assert response.status_code == 422


def test_list_usage_skip_negative_rejected(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"skip": -1},
    )
    assert response.status_code == 422


def test_list_usage_filter_by_epoch_seconds(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    older = datetime(2025, 8, 1, 0, 0, tzinfo=UTC)
    newer = datetime(2025, 8, 2, 0, 0, tzinfo=UTC)
    midpoint_epoch = int((older + timedelta(hours=12)).timestamp())
    _make_log(db_session, user_id="epoch-user", timestamp=older)
    _make_log(db_session, user_id="epoch-user", timestamp=newer)
    db_session.commit()

    response = client.get(
        USAGE_PATH,
        headers=master_key_header,
        params={"start_date": midpoint_epoch},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["timestamp"] == newer.isoformat()
