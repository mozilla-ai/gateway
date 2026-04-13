from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from gateway.core.config import GatewayConfig
from gateway.db import APIKey, User
from gateway.main import create_app


def test_create_app_bootstraps_first_api_key(tmp_path: Path) -> None:
    database_path = tmp_path / "bootstrap.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")
    app = create_app(config)

    with TestClient(app):
        pass

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        keys = db.query(APIKey).all()
        users = db.query(User).all()

    engine.dispose()

    assert len(keys) == 1
    assert keys[0].key_name == "bootstrap"
    assert keys[0].metadata_.get("bootstrap") is True
    assert len(users) == 1
    assert users[0].user_id == keys[0].user_id


def test_create_app_does_not_create_second_bootstrap_key(tmp_path: Path) -> None:
    database_path = tmp_path / "bootstrap-once.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")

    app = create_app(config)
    with TestClient(app):
        pass
    # Second startup should not create another key
    app_again = create_app(config)
    with TestClient(app_again):
        pass

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        key_count = db.query(APIKey).count()
    engine.dispose()

    assert key_count == 1


def test_create_app_skips_bootstrap_when_disabled(tmp_path: Path) -> None:
    database_path = tmp_path / "no-bootstrap.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}", bootstrap_api_key=False)

    app = create_app(config)
    with TestClient(app):
        pass

    engine = create_engine(config.database_url)
    with Session(engine) as db:
        key_count = db.query(APIKey).count()
        user_count = db.query(User).count()
    engine.dispose()

    assert key_count == 0
    assert user_count == 0
