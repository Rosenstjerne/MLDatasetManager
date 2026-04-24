from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from mldatasetmanager.adapters.registry import get_adapter
from mldatasetmanager.core.models import (
    Dataset,
    DatasetTask,
    ImageAsset,
    MultiPolygon,
    OrientedBBox,
    RLEMask,
)
from mldatasetmanager.reports import SplitReport, ValidationReport
from mldatasetmanager.validation.validators import validate_dataset

SPLIT_NAMES = ("train", "val", "test")


def resplit_dataset(
    source: Path,
    output: Path,
    source_format: str,
    target_format: str,
    task: str,
    ratios: list[float],
    seed: int,
    stratify: str = "none",
    options: dict | None = None,
) -> SplitReport:
    options = options or {}
    overwrite = bool(options.get("overwrite", False))
    validation = ValidationReport(dataset_path=str(source))
    try:
        normalized_ratios = _normalize_ratios(ratios)
    except ValueError as exc:
        normalized_ratios = [0.0, 0.0, 0.0]
        validation.add("error", "SPLIT_RATIOS_INVALID", str(exc))
    report = SplitReport(
        source_path=str(source),
        output_path=str(output),
        source_format=source_format,
        target_format=target_format,
        task=task,
        ratios=dict(zip(SPLIT_NAMES, normalized_ratios, strict=True)),
        seed=seed,
        stratify=stratify,
        validation=validation,
    )

    if validation.has_errors:
        return report
    if task not in {"detection", "segmentation", "obb"}:
        validation.add(
            "error",
            "UNSUPPORTED_TASK",
            "MVP split task must be 'detection', 'segmentation', or 'obb'",
        )
        return report
    if stratify not in {"none", "class"}:
        validation.add(
            "error",
            "UNSUPPORTED_STRATIFY",
            "Stratify must be 'none' or 'class'",
        )
        return report

    source_adapter = get_adapter(source_format)
    structure_report = source_adapter.validate_structure(source)
    validation.diagnostics.extend(structure_report.diagnostics)
    if structure_report.has_errors:
        return report
    target_task = _task_from_cli(task)
    try:
        dataset = source_adapter.read(source, {"task": target_task.value})
    except (KeyError, TypeError, ValueError, OSError, ValidationError) as exc:
        validation.add(
            "error",
            "DATASET_READ_FAILED",
            f"Failed to read {source_format} dataset: {exc}",
            file_path=str(source),
        )
        return report

    dataset_report = validate_dataset(dataset)
    validation.diagnostics.extend(dataset_report.diagnostics)
    if target_task == DatasetTask.ORIENTED_DETECTION:
        for annotation in dataset.annotations:
            if not _can_export_as_obb(annotation.geometry):
                validation.add(
                    "error",
                    "OBB_QUADRILATERAL_REQUIRED",
                    "OBB split export requires bbox, OBB, or quadrilateral polygon annotations",
                    annotation_id=annotation.id,
                )
    if validation.has_errors:
        return report

    split_dataset = _assign_splits(dataset, normalized_ratios, seed, stratify)
    if output.exists():
        if not overwrite:
            validation.add(
                "error",
                "OUTPUT_EXISTS",
                "Output path already exists; use overwrite to replace it",
                file_path=str(output),
            )
            return report
        shutil.rmtree(output)

    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        files_written = get_adapter(target_format).write(
            split_dataset,
            staging,
            {"task": target_task.value, "split_output": True},
        )
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report.files_written = files_written
    report.summary = {
        "images": len(split_dataset.images),
        "annotations": len(split_dataset.annotations),
        "categories": len(split_dataset.classes),
        "splits": {name: len(ids) for name, ids in split_dataset.splits.items()},
    }
    return report


def _normalize_ratios(ratios: list[float]) -> list[float]:
    if len(ratios) != 3:
        raise ValueError("Split ratios must contain exactly three values: train,val,test")
    if any(value < 0 for value in ratios):
        raise ValueError("Split ratios must be non-negative")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return [value / total for value in ratios]


def _assign_splits(
    dataset: Dataset,
    ratios: list[float],
    seed: int,
    stratify: str,
) -> Dataset:
    rng = random.Random(seed)
    ordered_images = list(dataset.images)
    if stratify == "class":
        ordered_images = _class_interleaved_images(dataset, rng)
    else:
        rng.shuffle(ordered_images)

    counts = _target_counts(len(ordered_images), ratios)
    split_by_image_id: dict[int | str, str] = {}
    cursor = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for image in ordered_images[cursor : cursor + count]:
            split_by_image_id[image.id] = split_name
        cursor += count

    images = [
        ImageAsset(
            id=image.id,
            path=image.path,
            width=image.width,
            height=image.height,
            checksum=image.checksum,
            split=split_by_image_id[image.id],
            metadata=image.metadata,
        )
        for image in dataset.images
    ]
    splits = {
        split_name: [image.id for image in images if image.split == split_name]
        for split_name in SPLIT_NAMES
    }
    return Dataset(
        id=dataset.id,
        name=dataset.name,
        dataset_type=dataset.dataset_type,
        root=dataset.root,
        classes=dataset.classes,
        images=images,
        annotations=dataset.annotations,
        splits=splits,
        metadata={**dataset.metadata, "resplit": {"seed": seed, "ratios": ratios}},
        provenance={**dataset.provenance, "operation": "resplit"},
    )


def _target_counts(total: int, ratios: list[float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(raw_counts)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _class_interleaved_images(dataset: Dataset, rng: random.Random) -> list[ImageAsset]:
    annotations_by_image: dict[int | str, list[int]] = defaultdict(list)
    for annotation in dataset.annotations:
        annotations_by_image[annotation.image_id].append(annotation.category_id)

    buckets: dict[int, list[ImageAsset]] = defaultdict(list)
    for image in dataset.images:
        category_counts = Counter(annotations_by_image.get(image.id, []))
        primary_category = category_counts.most_common(1)[0][0] if category_counts else -1
        buckets[primary_category].append(image)

    for images in buckets.values():
        rng.shuffle(images)

    ordered: list[ImageAsset] = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop())
    return ordered


def _task_from_cli(task: str) -> DatasetTask:
    if task == "segmentation":
        return DatasetTask.INSTANCE_SEGMENTATION
    if task == "obb":
        return DatasetTask.ORIENTED_DETECTION
    return DatasetTask.DETECTION


def _can_export_as_obb(geometry) -> bool:
    if isinstance(geometry, OrientedBBox):
        return True
    if isinstance(geometry, RLEMask):
        return False
    if isinstance(geometry, MultiPolygon):
        return len(geometry.polygons) == 1 and len(geometry.polygons[0].points) == 4
    return True
