from pathlib import Path

from pydantic import ValidationError

from mldatasetmanager.adapters.registry import get_adapter
from mldatasetmanager.conversion.pipeline import (
    convert_coco_to_yolo,
    convert_yolo_to_coco,
    convert_yolo_to_yolo,
)
from mldatasetmanager.core.models import Dataset
from mldatasetmanager.merging.merger import merge_datasets as merge_dataset_sources
from mldatasetmanager.reports import ConversionReport, MergeReport, SplitReport, ValidationReport
from mldatasetmanager.splitting.splitter import resplit_dataset as resplit_dataset_source
from mldatasetmanager.validation.validators import validate_dataset as validate_loaded_dataset


def import_dataset(path: str | Path, format: str, options: dict | None = None) -> Dataset:
    adapter = get_adapter(format)
    return adapter.read(Path(path), options or {})


def validate_dataset(
    dataset_or_path: Dataset | str | Path,
    format: str | None = None,
    options: dict | None = None,
) -> ValidationReport:
    if isinstance(dataset_or_path, Dataset):
        return validate_loaded_dataset(dataset_or_path)
    if format is None:
        raise ValueError("format is required when validating a path")
    adapter = get_adapter(format)
    structure_report = adapter.validate_structure(Path(dataset_or_path))
    if structure_report.has_errors:
        return structure_report
    try:
        dataset = adapter.read(Path(dataset_or_path), options or {})
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        structure_report.add("error", "DATASET_READ_FAILED", f"Failed to read dataset: {exc}")
        return structure_report
    return validate_loaded_dataset(dataset)


def convert_dataset(
    source: str | Path,
    output: str | Path,
    source_format: str,
    target_format: str,
    task: str,
    options: dict | None = None,
) -> ConversionReport:
    source_format = source_format.lower()
    target_format = target_format.lower()
    if source_format == "coco" and target_format == "yolo":
        return convert_coco_to_yolo(Path(source), Path(output), task=task, options=options or {})
    if source_format == "yolo" and target_format == "coco":
        return convert_yolo_to_coco(Path(source), Path(output), task=task, options=options or {})
    if source_format == "yolo" and target_format == "yolo":
        return convert_yolo_to_yolo(Path(source), Path(output), task=task, options=options or {})
    raise ValueError("MVP conversion support is limited to COCO <-> YOLO and YOLO -> YOLO")


def merge_datasets(
    sources: list[str | Path],
    output: str | Path,
    source_formats: list[str],
    target_format: str,
    task: str,
    options: dict | None = None,
) -> MergeReport:
    return merge_dataset_sources(
        [Path(source) for source in sources],
        Path(output),
        source_formats=source_formats,
        target_format=target_format,
        task=task,
        options=options or {},
    )


def resplit_dataset(
    source: str | Path,
    output: str | Path,
    source_format: str,
    target_format: str,
    task: str,
    ratios: list[float],
    seed: int = 42,
    stratify: str = "none",
    options: dict | None = None,
) -> SplitReport:
    return resplit_dataset_source(
        Path(source),
        Path(output),
        source_format=source_format,
        target_format=target_format,
        task=task,
        ratios=ratios,
        seed=seed,
        stratify=stratify,
        options=options or {},
    )
