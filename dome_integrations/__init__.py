# Local copy of vijil_dome telemetry integration for Darwin compatibility
# This can be removed once vijil-dome is published with the telemetry module
from .telemetry import instrument_for_darwin, darwin_trace

__all__ = ["instrument_for_darwin", "darwin_trace"]
