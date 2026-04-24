from __future__ import annotations

import shutil
from pathlib import Path

from mldatasetmanager.adapters.coco import CocoAdapter, try_read_coco
from mldatasetmanager.adapters.yolo import YoloAdapter, try_read_yolo
from mldatasetmanager.core.models import DatasetTask, MultiPolygon, RLEMask
from mldatasetmanager.reports import ConversionReport, ValidationReport
from mldatasetmanager.validation.validators import validate_dataset


def convert_coco_to_yolo(
    source: Path,
    output: Path,
    task: str,
    options: dict | None = None,
) -> ConversionReport:
    options = options or {}
    overwrite = bool(options.get("overwrite", False))
    report = ConversionReport(
        source_path=str(source),
        output_path=str(output),
        source_format="coco",
        target_format="yolo",
        task=task,
        validation=ValidationReport(dataset_path=str(source)),
    )
    if task not in {"detection", "segmentation"}:
        report.validation.add(
            "error",
            "UNSUPPORTED_TASK",
            "MVP conversion task must be 'detection' or 'segmentation'",
        )
        return report
    dataset, structure_report = try_read_coco(source)
    if dataset is None:
        report.validation = structure_report
        return report

    validation_report = validate_dataset(dataset)
    target_task = (
        DatasetTask.INSTANCE_SEGMENTATION if task == "segmentation" else DatasetTask.DETECTION
    )
    if target_task == DatasetTask.INSTANCE_SEGMENTATION:
        for annotation in dataset.annotations:
            if isinstance(annotation.geometry, RLEMask):
                validation_report.add(
                    "error",
                    "RLE_SEGMENTATION_UNSUPPORTED",
                    "RLE segmentation cannot be exported to YOLO segmentation in the MVP",
                    annotation_id=annotation.id,
                )
            elif not isinstance(annotation.geometry, MultiPolygon):
                validation_report.add(
                    "error",
                    "SEGMENTATION_REQUIRED",
                    "YOLO segmentation export requires polygon segmentation annotations",
                    annotation_id=annotation.id,
                )
    report.validation = validation_report
    if validation_report.has_errors:
        return report

    if output.exists():
        if not overwrite:
            report.validation.add(
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
        files_written = YoloAdapter().write(dataset, staging, {"task": target_task.value})
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    report.files_written = files_written
    report.summary = {
        "images": len(dataset.images),
        "annotations": len(dataset.annotations),
        "categories": len(dataset.classes),
    }
    return report


def convert_yolo_to_coco(
    source: Path,
    output: Path,
    task: str,
    options: dict | None = None,
) -> ConversionReport:
    options = options or {}
    overwrite = bool(options.get("overwrite", False))
    report = ConversionReport(
        source_path=str(source),
        output_path=str(output),
        source_format="yolo",
        target_format="coco",
        task=task,
        validation=ValidationReport(dataset_path=str(source)),
    )
    if task not in {"detection", "segmentation"}:
        report.validation.add(
            "error",
            "UNSUPPORTED_TASK",
            "MVP conversion task must be 'detection' or 'segmentation'",
        )
        return report

    target_task = (
        DatasetTask.INSTANCE_SEGMENTATION if task == "segmentation" else DatasetTask.DETECTION
    )
    dataset, structure_report = try_read_yolo(source, {"task": target_task.value})
    if dataset is None:
        report.validation = structure_report
        return report

    validation_report = validate_dataset(dataset)
    if target_task == DatasetTask.INSTANCE_SEGMENTATION:
        for annotation in dataset.annotations:
            if not isinstance(annotation.geometry, MultiPolygon):
                validation_report.add(
                    "error",
                    "SEGMENTATION_REQUIRED",
                    "COCO segmentation export requires YOLO segmentation label rows",
                    annotation_id=annotation.id,
                )
    report.validation = validation_report
    if validation_report.has_errors:
        return report

    if output.exists():
        if not overwrite:
            report.validation.add(
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
        files_written = CocoAdapter().write(dataset, staging, {"task": target_task.value})
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    report.files_written = files_written
    report.summary = {
        "images": len(dataset.images),
        "annotations": len(dataset.annotations),
        "categories": len(dataset.classes),
    }
    return report
