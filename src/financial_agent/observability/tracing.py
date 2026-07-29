"""Tracing helpers.

`traced_span` is the primitive every Tool handler wraps itself in (see
`financial_agent.agent.tools.base.BaseTool.arun`) to record execution time
and success/failure as a structured log event — this satisfies the
"tempo de execução das Tools" observability requirement without a hard
dependency on any specific backend.

If the optional `opentelemetry-sdk` / `opentelemetry-instrumentation-fastapi`
packages are installed and `OTEL_EXPORTER_OTLP_ENDPOINT` is set, spans are
*also* emitted as real OpenTelemetry spans, so this project can be wired
into Jaeger/Tempo/X-Ray/etc without changing call sites. Absent that
dependency, tracing degrades gracefully to structured logs only.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from financial_agent.observability.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised only when the optional extra is installed
    from opentelemetry import trace as _otel_trace

    _tracer: Any | None = _otel_trace.get_tracer("financial_agent")
except ImportError:  # pragma: no cover
    _tracer = None


@contextmanager
def traced_span(name: str, **attributes: object) -> Generator[None, None, None]:
    """Context manager that logs (and optionally OTel-traces) span duration.

    Usage:
        with traced_span("tool.get_next_best_action", user_id=user_id):
            ...
    """
    start = time.perf_counter()
    otel_span_cm = _tracer.start_as_current_span(name) if _tracer is not None else None
    if otel_span_cm is not None:
        otel_span_cm.__enter__()

    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "span_finished",
            span=name,
            status=status,
            duration_ms=duration_ms,
            **attributes,
        )
        if otel_span_cm is not None:
            otel_span_cm.__exit__(None, None, None)
