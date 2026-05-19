"""URL safety checks for inline MCP server endpoints.

Two concerns:

1. **SSRF.** A request body could otherwise point the gateway at internal services
   (AWS IMDS, internal databases, cluster-local endpoints) and exfiltrate data via
   the tool-list response. We resolve the host and reject private, link-local, and
   reserved IP ranges. Loopback is allowed by default since it's genuinely useful
   for local dev and same-host sidecar deployments — set ``GATEWAY_MCP_ALLOW_LOOPBACK=false``
   to disable.

2. **TLS-with-token.** A bearer token over an ``http://`` URL is exfiltratable by
   any on-path observer. If the caller provides a token, the URL must be HTTPS.

These checks are intentionally conservative: DNS rebinding can defeat host-based
allowlists. Production deployments should also enforce egress policy at the
network layer.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse


class UnsafeURLError(ValueError):
    """Raised when an MCP server URL is rejected by the safety checks."""


def _allow_loopback() -> bool:
    return os.environ.get("GATEWAY_MCP_ALLOW_LOOPBACK", "true").lower() not in {"0", "false", "no"}


def _allow_private_hosts() -> bool:
    return os.environ.get("GATEWAY_MCP_ALLOW_PRIVATE_HOSTS", "false").lower() in {"1", "true", "yes"}


def _resolve_all(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        try:
            out.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return out


def validate_mcp_url(url: str, *, has_authorization_token: bool) -> None:
    """Reject URLs that are unsafe for the gateway to fetch.

    Raises :class:`UnsafeURLError` on rejection. Returns ``None`` on accept.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"MCP server URL must use http or https, got {scheme!r}")
    if scheme == "http" and has_authorization_token:
        raise UnsafeURLError("MCP server URL must use https when an authorization_token is set")

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("MCP server URL must include a hostname")

    if _allow_private_hosts():
        return

    try:
        literal = ipaddress.ip_address(host)
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal]
    except ValueError:
        addresses = _resolve_all(host)
        if not addresses:
            # Couldn't resolve — allow through; the request will fail at fetch time.
            # Don't leak resolution success/failure via reject behaviour.
            return

    for addr in addresses:
        if addr.is_loopback and _allow_loopback():
            continue
        reason = _blocked_reason(addr)
        if reason is not None:
            raise UnsafeURLError(
                f"MCP server host {host!r} resolves to {addr} which is {reason}; "
                "rejecting to prevent SSRF. Set GATEWAY_MCP_ALLOW_PRIVATE_HOSTS=true to override."
            )


def _blocked_reason(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    # Order matters: is_private returns True for unspecified/loopback/link-local too,
    # so more specific labels go first to produce useful error messages.
    if addr.is_unspecified:
        return "unspecified (0.0.0.0/::)"
    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link-local"
    if addr.is_multicast:
        return "multicast"
    if addr.is_private:
        return "in a private range (RFC 1918 / ULA)"
    if addr.is_reserved:
        return "in a reserved range"
    return None
