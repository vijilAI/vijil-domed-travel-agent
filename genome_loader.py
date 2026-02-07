"""Genome Loader - Dynamic mutation loading for Darwin evolution.

This module enables the domed travel agent to load mutations from a file
without requiring redeployment. Darwin writes mutation checkpoints to a
JSON file, and the agent reloads it periodically or on each request.

File Format (genome.json):
{
    "version": 1,
    "created_at": "2026-02-06T12:00:00Z",
    "system_prompt": "You are a secure travel assistant...",
    "dome_config": {
        "security_threshold": 0.7,
        "moderation_threshold": 0.5,
        ...
    }
}

Usage:
    # Agent loads genome on startup and periodically refreshes
    loader = GenomeLoader(os.environ.get("GENOME_PATH"))

    # Get current mutation (reloads if file changed)
    genome = loader.get_current()
    system_prompt = genome.system_prompt
    dome_overrides = genome.dome_config
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenomeMutation:
    """Represents a loaded genome mutation checkpoint."""

    version: int = 0
    created_at: str = ""
    system_prompt: str | None = None
    dome_config: dict[str, Any] = field(default_factory=dict)

    # Metadata
    source_file: str = ""
    loaded_at: float = 0.0

    @classmethod
    def from_dict(cls, data: dict, source_file: str = "") -> "GenomeMutation":
        """Create GenomeMutation from dictionary."""
        return cls(
            version=data.get("version", 0),
            created_at=data.get("created_at", ""),
            system_prompt=data.get("system_prompt"),
            dome_config=data.get("dome_config", {}),
            source_file=source_file,
            loaded_at=time.time(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "system_prompt": self.system_prompt,
            "dome_config": self.dome_config,
        }


class GenomeLoader:
    """Loads and caches genome mutations from a file.

    Supports hot-reload: checks file modification time and reloads
    if the file has changed since last load.
    """

    def __init__(
        self,
        genome_path: str | None = None,
        reload_interval_seconds: float = 5.0,
    ):
        """Initialize the genome loader.

        Args:
            genome_path: Path to genome.json file. If None, returns defaults.
            reload_interval_seconds: Minimum time between reload checks.
        """
        self.genome_path = Path(genome_path) if genome_path else None
        self.reload_interval = reload_interval_seconds

        self._cached_genome: GenomeMutation | None = None
        self._last_check_time: float = 0.0
        self._last_mtime: float = 0.0

    def get_current(self) -> GenomeMutation:
        """Get the current genome mutation, reloading if file changed.

        Returns:
            GenomeMutation with current values (or defaults if no file).
        """
        if self.genome_path is None:
            # No genome file configured, return empty defaults
            return GenomeMutation()

        now = time.time()

        # Rate-limit file checks
        if now - self._last_check_time < self.reload_interval:
            if self._cached_genome is not None:
                return self._cached_genome

        self._last_check_time = now

        # Check if file exists and has been modified
        try:
            if not self.genome_path.exists():
                logger.debug(f"Genome file not found: {self.genome_path}")
                return self._cached_genome or GenomeMutation()

            mtime = self.genome_path.stat().st_mtime

            # Reload if file modified or no cache
            if mtime != self._last_mtime or self._cached_genome is None:
                self._cached_genome = self._load_genome()
                self._last_mtime = mtime
                logger.info(
                    f"Loaded genome v{self._cached_genome.version} "
                    f"from {self.genome_path}"
                )

        except Exception as e:
            logger.warning(f"Error checking genome file: {e}")
            if self._cached_genome is None:
                return GenomeMutation()

        return self._cached_genome

    def _load_genome(self) -> GenomeMutation:
        """Load genome from file."""
        try:
            with open(self.genome_path, "r") as f:
                data = json.load(f)
            return GenomeMutation.from_dict(data, str(self.genome_path))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in genome file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load genome file: {e}")
            raise

    def force_reload(self) -> GenomeMutation:
        """Force reload the genome file, ignoring cache."""
        self._last_mtime = 0.0
        self._last_check_time = 0.0
        return self.get_current()


# Global loader instance (initialized by agent on startup)
_loader: GenomeLoader | None = None


def init_genome_loader(genome_path: str | None = None) -> GenomeLoader:
    """Initialize the global genome loader.

    Args:
        genome_path: Path to genome.json. Defaults to GENOME_PATH env var.

    Returns:
        Initialized GenomeLoader instance.
    """
    global _loader
    path = genome_path or os.environ.get("GENOME_PATH")
    _loader = GenomeLoader(path)

    if path:
        logger.info(f"Genome loader initialized: {path}")
    else:
        logger.info("Genome loader initialized (no file, using defaults)")

    return _loader


def get_genome_loader() -> GenomeLoader:
    """Get the global genome loader, initializing if needed."""
    global _loader
    if _loader is None:
        _loader = init_genome_loader()
    return _loader


def get_current_genome() -> GenomeMutation:
    """Convenience function to get current genome mutation."""
    return get_genome_loader().get_current()
