from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mldatasetmanager.core.models import Dataset
from mldatasetmanager.reports import ValidationReport


@dataclass(frozen=True)
class AdapterCapabilities:
    tasks: set[str] = field(default_factory=set)
    geometries: set[str] = field(default_factory=set)
    coordinate_system: str = "pixel_absolute"


@dataclass(frozen=True)
class DetectionResult:
    matched: bool
    confidence: float
    reason: str


class DatasetAdapter(Protocol):
    format_name: str
    capabilities: AdapterCapabilities

    def detect(self, path: Path) -> DetectionResult: ...

    def read(self, path: Path, options: dict | None = None) -> Dataset: ...

    def write(self, dataset: Dataset, path: Path, options: dict | None = None) -> int: ...

    def validate_structure(self, path: Path) -> ValidationReport: ...
