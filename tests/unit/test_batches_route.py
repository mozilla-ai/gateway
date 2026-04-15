"""Unit tests for batch route Pydantic request models."""

import pytest
from pydantic import ValidationError

from gateway.api.routes.batches import BatchRequestItem, CreateBatchRequest


class TestBatchRequestItem:
    def test_valid_item(self) -> None:
        item = BatchRequestItem(
            custom_id="req-1",
            body={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
        )
        assert item.custom_id == "req-1"
        assert "messages" in item.body

    def test_missing_custom_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="custom_id"):
            BatchRequestItem(body={"messages": []})  # type: ignore[call-arg]


class TestCreateBatchRequest:
    def test_valid_request(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[
                BatchRequestItem(
                    custom_id="req-1",
                    body={"messages": [{"role": "user", "content": "Hello"}]},
                ),
            ],
        )
        assert request.model == "openai:gpt-4o-mini"
        assert len(request.requests) == 1
        assert request.completion_window == "24h"
        assert request.metadata is None

    def test_empty_requests_rejected(self) -> None:
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            CreateBatchRequest(model="openai:gpt-4o-mini", requests=[])

    def test_too_many_requests_rejected(self) -> None:
        items = [BatchRequestItem(custom_id=f"req-{i}", body={}) for i in range(10_001)]
        with pytest.raises(ValidationError, match="List should have at most 10000 items"):
            CreateBatchRequest(model="openai:gpt-4o-mini", requests=items)

    def test_missing_model_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model"):
            CreateBatchRequest(
                requests=[BatchRequestItem(custom_id="req-1", body={})],
            )  # type: ignore[call-arg]

    def test_optional_metadata(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[BatchRequestItem(custom_id="req-1", body={})],
            metadata={"team": "ml-ops"},
        )
        assert request.metadata == {"team": "ml-ops"}

    def test_custom_completion_window(self) -> None:
        request = CreateBatchRequest(
            model="openai:gpt-4o-mini",
            requests=[BatchRequestItem(custom_id="req-1", body={})],
            completion_window="48h",
        )
        assert request.completion_window == "48h"
