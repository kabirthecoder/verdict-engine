"""One shared HTTP client for the supply-chain tools: timeouts, retries, user agent."""

from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

UA = "verdict-engine/0.1 (+https://github.com/kabirthecoder/verdict-engine)"


class UpstreamError(RuntimeError):
    pass


_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=httpx.Timeout(20.0, connect=10.0), headers={"User-Agent": UA}
        )
    return _client


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, UpstreamError) and getattr(exc, "status", 0) >= 500


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TransportError, UpstreamError)),
    reraise=True,
)
def get_json(
    url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> Any:
    r = client().get(url, params=params, headers=headers)
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        err = UpstreamError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
        err.status = r.status_code  # type: ignore[attr-defined]
        if r.status_code >= 500:
            raise err
        raise err from None
    return r.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TransportError, UpstreamError)),
    reraise=True,
)
def post_json(url: str, body: dict[str, Any]) -> Any:
    r = client().post(url, json=body)
    if r.status_code >= 400:
        err = UpstreamError(f"POST {url} -> {r.status_code}: {r.text[:200]}")
        err.status = r.status_code  # type: ignore[attr-defined]
        raise err
    return r.json()


def github_headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h
