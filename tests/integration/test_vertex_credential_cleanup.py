"""Tests for request-local Vertex credential handling."""

import os
from unittest.mock import sentinel

import pytest
from fastapi import HTTPException

from gateway.auth.vertex_auth import setup_vertex_environment


def test_setup_vertex_environment_returns_kwargs_without_env_mutation() -> None:
    """Credentials are resolved into client kwargs without mutating process env."""
    original_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    original_location = os.environ.get("GOOGLE_CLOUD_LOCATION")

    os.environ["GOOGLE_CLOUD_PROJECT"] = "existing-project"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "europe-west1"

    try:
        from gateway.auth import vertex_auth

        def _fake_from_info(info: dict[str, object]) -> object:
            assert info["project_id"] == "config-project"
            return sentinel.credentials

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                vertex_auth.service_account.Credentials,
                "from_service_account_info",
                _fake_from_info,
            )
            kwargs = setup_vertex_environment(
                credentials={"project_id": "config-project", "private_key": "unused"},
                location="us-central1",
            )

        assert kwargs["credentials"] is sentinel.credentials
        assert kwargs["project"] == "config-project"
        assert kwargs["location"] == "us-central1"
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == "existing-project"
        assert os.environ["GOOGLE_CLOUD_LOCATION"] == "europe-west1"
    finally:
        if original_project is None:
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        else:
            os.environ["GOOGLE_CLOUD_PROJECT"] = original_project

        if original_location is None:
            os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
        else:
            os.environ["GOOGLE_CLOUD_LOCATION"] = original_location


def test_setup_vertex_environment_uses_env_fallbacks() -> None:
    """Project and location fall back to env when explicit values are missing."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        mp.setenv("GOOGLE_CLOUD_LOCATION", "env-location")

        kwargs = setup_vertex_environment(credentials=None)

    assert kwargs == {"project": "env-project", "location": "env-location"}


def test_setup_vertex_environment_requires_project_resolution() -> None:
    """Project resolution failure raises a clear 400 error."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        mp.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        with pytest.raises(HTTPException) as exc:
            setup_vertex_environment(credentials=None)

    assert exc.value.status_code == 400
    assert "Could not resolve GCP project ID" in str(exc.value.detail)
