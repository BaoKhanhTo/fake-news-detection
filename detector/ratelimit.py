"""Tiny per-client rate limiter backed by django.core.cache.

Uses a fixed-window counter — good enough for a single-process dev server
and unintrusive for production behind a reverse proxy that already
does proper rate limiting. Swap the cache backend (e.g. Redis) via
CACHES in settings.py if you need multi-process correctness.
"""

from django.core.cache import cache


def _client_id(request):
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "anonymous")


def check(request, *, scope, max_requests, window_seconds):
    """Return (allowed: bool, retry_after_seconds: int).

    `scope` namespaces the counter so different endpoints don't share a bucket.
    """
    key = f"ratelimit:{scope}:{_client_id(request)}"
    count = cache.get(key, 0)
    if count >= max_requests:
        return False, window_seconds
    # set timeout only on first hit so the window doesn't keep sliding
    if count == 0:
        cache.set(key, 1, timeout=window_seconds)
    else:
        try:
            cache.incr(key)
        except ValueError:
            cache.set(key, count + 1, timeout=window_seconds)
    return True, 0
