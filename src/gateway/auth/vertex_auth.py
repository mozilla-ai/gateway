"""Helpers for preparing Vertex AI client kwargs safely per request."""

import json
import os
from typing import Any

from fastapi import HTTPException, status
from google.oauth2 import service_account


def setup_vertex_environment(
    credentials: str | dict[str, Any] | None = None,
    project: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Build request-local Vertex AI client kwargs for any-llm.

    The Vertex provider passes these kwargs directly to `google.genai.Client`,
    so this function avoids process-global environment mutation.

    Args:
        credentials: Path to service account JSON file, JSON string, or dict
        project: Optional GCP project ID (extracted from credentials if not provided)
        location: Optional GCP location (default: "us-central1")

    Returns:
        Dict of kwargs for provider initialization

    Raises:
        HTTPException: If credentials cannot be loaded or project cannot be determined

    """
    kwargs: dict[str, Any] = {}

    if credentials is not None:
        json_obj: dict[str, Any] | None = None

        if isinstance(credentials, str):
            try:
                if os.path.exists(credentials):
                    with open(credentials, encoding="utf-8") as f:
                        json_obj = json.load(f)
                    kwargs["credentials"] = service_account.Credentials.from_service_account_file(credentials)
                else:
                    json_obj = json.loads(credentials)
                    kwargs["credentials"] = service_account.Credentials.from_service_account_info(json_obj)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to load vertex credentials",
                ) from e
        elif isinstance(credentials, dict):
            json_obj = credentials
            try:
                kwargs["credentials"] = service_account.Credentials.from_service_account_info(credentials)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to load vertex credentials",
                ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid credentials type: {type(credentials)}",
            )

        if project is None and json_obj:
            project = json_obj.get("project_id")

    if project:
        kwargs["project"] = project
    elif "GOOGLE_CLOUD_PROJECT" not in os.environ:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not resolve GCP project ID from credentials or configuration",
        )
    else:
        kwargs["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]

    if location:
        kwargs["location"] = location
    elif "GOOGLE_CLOUD_LOCATION" not in os.environ:
        kwargs["location"] = "us-central1"
    else:
        kwargs["location"] = os.environ["GOOGLE_CLOUD_LOCATION"]

    return kwargs
