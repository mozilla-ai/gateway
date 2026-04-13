from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.version import __version__

router = APIRouter(prefix="/health", tags=["health"])


async def _check_platform_reachability(config: GatewayConfig) -> bool:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        return False

    health_path = config.platform.get("health_path", "/utils/health-check/")
    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    health_url = f"{platform_base_url.rstrip('/')}/{health_path.lstrip('/')}"

    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as client:
            response = await client.get(health_url)
        return response.status_code < 500
    except (httpx.TimeoutException, httpx.NetworkError):
        return False


@router.get("")
async def health_check(config: GatewayConfig = Depends(get_config)) -> dict[str, str]:
    """General health check endpoint.

    Returns basic health status. For infrastructure monitoring,
    use /health/readiness or /health/liveness instead.
    """
    payload: dict[str, str] = {"status": "healthy"}
    if config.is_platform_mode:
        payload["mode"] = "platform"
        payload["platform_reachable"] = "yes" if await _check_platform_reachability(config) else "no"
    return payload


@router.get("/liveness")
async def health_liveness() -> str:
    """Liveness probe endpoint.

    Simple check to verify the process is alive and responding.
    Used by Kubernetes/container orchestrators for liveness probes.

    Returns:
        Plain text "I'm alive!" message

    """
    return "I'm alive!"


@router.get("/readiness")
async def health_readiness(
    config: GatewayConfig = Depends(get_config),
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)] = None,
) -> dict[str, Any]:
    """Readiness probe endpoint.

    Checks if the gateway is ready to serve requests by validating:
    - Database connectivity
    - Service availability

    Used by Kubernetes/container orchestrators for readiness probes.
    Returns HTTP 503 if any dependency is unavailable.

    Returns:
        dict: Status object with health details

    Raises:
        HTTPException: 503 if service is not ready

    """
    if config.is_platform_mode:
        platform_reachable = await _check_platform_reachability(config)
        if not platform_reachable:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "mode": "platform",
                    "platform": "unavailable",
                    "version": __version__,
                },
            )
        return {
            "status": "healthy",
            "mode": "platform",
            "platform": "connected",
            "version": __version__,
        }

    if db is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "database": "unavailable", "version": __version__},
        )

    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error("Database connectivity check failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "unavailable",
                "version": __version__,
            },
        ) from e
    return {
        "status": "healthy",
        "database": db_status,
        "version": __version__,
    }
