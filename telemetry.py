"""OpenTelemetry setup for Darwin-compatible telemetry emission.

This module configures OTEL exporters to send traces and metrics to the
Vijil observability stack (Tempo/Mimir via OTEL Collector).

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint
        Default: http://otel-collector.vijil-telemetry.svc.cluster.local:4318
    OTEL_SERVICE_NAME: Service name for traces
        Default: vijil-domed-travel-agent
    TEAM_ID: Team ID for multi-tenant filtering (optional)
"""

import os
import logging
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

logger = logging.getLogger(__name__)


# Default OTEL collector endpoint (inside Kind cluster)
DEFAULT_OTLP_ENDPOINT = "http://otel-collector.vijil-telemetry.svc.cluster.local:4318"

# Service name for this agent
SERVICE_NAME_VALUE = "vijil-domed-travel-agent"


def setup_telemetry(
    otlp_endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
) -> tuple[trace.Tracer, metrics.Meter]:
    """Set up OpenTelemetry tracing and metrics exporters.

    Configures OTLP HTTP exporters to send telemetry to the observability stack.
    The endpoint defaults to the in-cluster OTEL Collector service.

    Args:
        otlp_endpoint: OTLP collector endpoint (default: from env or cluster URL)
        service_name: Service name for resource attributes

    Returns:
        Tuple of (Tracer, Meter) for creating spans and metrics
    """
    # Get configuration from environment or defaults
    endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT
    )
    svc_name = service_name or os.environ.get("OTEL_SERVICE_NAME", SERVICE_NAME_VALUE)

    logger.info(f"Setting up OTEL telemetry: endpoint={endpoint}, service={svc_name}")

    # Create resource with service name
    resource = Resource.create({SERVICE_NAME: svc_name})

    # Set up tracing
    trace_exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Set up metrics
    metric_exporter = OTLPMetricExporter(
        endpoint=f"{endpoint}/v1/metrics",
    )
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=5000,  # Export every 5 seconds
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Create tracer and meter instances
    tracer = trace.get_tracer(svc_name)
    meter = metrics.get_meter(svc_name)

    logger.info("OTEL telemetry setup complete")

    return tracer, meter


def get_team_id() -> Optional[str]:
    """Get team ID from environment for multi-tenant filtering.

    Darwin uses team_id to filter detections when querying Tempo.

    Returns:
        Team ID string, or None if not configured
    """
    return os.environ.get("TEAM_ID")
