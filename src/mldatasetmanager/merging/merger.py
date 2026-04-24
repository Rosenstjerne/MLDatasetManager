from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import ValidationError

from mldatasetmanager.adapters.registry import get_adapter
from mldatasetmanager.core.models import (
    Annotation,
    Category,
    Dataset,
    DatasetTask,
    ImageAsset,
    MultiPolygon,
    OrientedBBox,
    RLEMask,
)
from mldatasetmanager.reports import MergeReport, ValidationReport
from mldatasetmanager.validation.validators import validate_dataset


def merge_datasets(
    sources: list[Path],
    output: Path,
    source_formats: list[str],
    target_format: str,
    task: str,
    options: dict | None = None,
) -> MergeReport:
    options = options or {}
    overwrite = bool(options.get("overwrite", False))
    class_policy = str(options.get("class_policy", "union"))
    validation = ValidationReport(dataset_path=";".join(str(source) for source in sources))
    report = MergeReport(
        source_paths=[str(source) for source in sources],
        output_path=str(output),
        source_formats=source_formats,
        target_format=target_format,
        task=task,
        class_policy=class_policy,
        validation=validation,
    )

    if task not in {"detection", "segmentation", "obb"}:
        validation.add(
            "error",
            "UNSUPPORTED_TASK",
            "MVP merge task must be 'detection', 'segmentation', or 'obb'",
        )
        return report
    if not sources:
        validation.add("error", "NO_SOURCES", "At least one source dataset is required")
        return report
    if len(source_formats) == 1 and len(sources) > 1:
        source_formats = source_formats * len(sources)
        report.source_formats = source_formats
    if len(source_formats) != len(sources):
        validation.add(
            "error",
            "FORMAT_COUNT_MISMATCH",
            "Number of source formats must match number of sources",
        )
        return report
    if class_policy not in {"union", "exact"}:
        validation.add(
            "error",
            "CLASS_POLICY_UNSUPPORTED",
            "Class policy must be 'union' or 'exact'",
        )
        return report

    target_task = _task_from_cli(task)
    datasets = _read_and_validate_sources(sources, source_formats, target_task, validation)
    if validation.has_errors:
        return report

    merged = _merge_loaded_datasets(datasets, target_task, class_policy, validation)
    if validation.has_errors or merged is None:
        return report

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
            merged,
            staging,
            {"task": target_task.value},
        )
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report.files_written = files_written
    report.summary = {
        "sources": len(sources),
        "images": len(merged.images),
        "annotations": len(merged.annotations),
        "categories": len(merged.classes),
    }
    return report


def _read_and_validate_sources(
    sources: list[Path],
    source_formats: list[str],
    target_task: DatasetTask,
    validation: ValidationReport,
) -> list[Dataset]:
    datasets = []
    for source, source_format in zip(sources, source_formats, strict=True):
        adapter = get_adapter(source_format)
        structure_report = adapter.validate_structure(source)
        validation.diagnostics.extend(structure_report.diagnostics)
        if structure_report.has_errors:
            continue
        try:
            dataset = adapter.read(source, {"task": target_task.value})
        except (KeyError, TypeError, ValueError, OSError, ValidationError) as exc:
            validation.add(
                "error",
                "DATASET_READ_FAILED",
                f"Failed to read {source_format} dataset: {exc}",
                file_path=str(source),
            )
            continue
        dataset_report = validate_dataset(dataset)
        validation.diagnostics.extend(dataset_report.diagnostics)
        datasets.append(dataset)
    return datasets


def _merge_loaded_datasets(
    datasets: list[Dataset],
    target_task: DatasetTask,
    class_policy: str,
    validation: ValidationReport,
) -> Dataset | None:
    if not datasets:
        validation.add("error", "NO_VALID_DATASETS", "No valid datasets were available to merge")
        return None

    merged_classes: list[Category] = []
    category_id_by_name: dict[str, int] = {}
    category_maps: list[dict[int, int]] = []
    for dataset_index, dataset in enumerate(datasets):
        category_map: dict[int, int] = {}
        for category in dataset.classes:
            existing_id = category_id_by_name.get(category.name)
            if existing_id is None:
                if class_policy == "exact" and dataset_index > 0:
                    validation.add(
                        "error",
                        "CLASS_MISSING_IN_MAIN",
                        f"Class '{category.name}' does not exist in the main dataset",
                        category_id=category.id,
                    )
                    continue
                existing_id = len(merged_classes)
                category_id_by_name[category.name] = existing_id
                merged_classes.append(
                    Category(
                        id=existing_id,
                        name=category.name,
                        supercategory=category.supercategory,
                        aliases=category.aliases,
                        metadata=category.metadata,
                    )
                )
            category_map[category.id] = existing_id
        category_maps.append(category_map)
    if validation.has_errors:
        return None

    merged_images: list[ImageAsset] = []
    merged_annotations: list[Annotation] = []
    used_file_names: set[str] = set()
    image_signature_to_id: dict[tuple[str, int, int], int] = {}
    annotation_signatures: set[tuple] = set()

    for dataset_index, dataset in enumerate(datasets):
        image_map: dict[int | str, int] = {}
        for image in dataset.images:
            signature = (str(image.path.resolve()), image.width, image.height)
            existing_image_id = image_signature_to_id.get(signature)
            if existing_image_id is not None:
                image_map[image.id] = existing_image_id
                continue
            merged_image_id = len(merged_images)
            target_file_name = _unique_file_name(image.path.name, used_file_names)
            image_signature_to_id[signature] = merged_image_id
            image_map[image.id] = merged_image_id
            merged_images.append(
                ImageAsset(
                    id=merged_image_id,
                    path=image.path,
                    width=image.width,
                    height=image.height,
                    checksum=image.checksum,
                    split=image.split or "train",
                    metadata={**image.metadata, "target_file_name": target_file_name},
                )
            )

        for annotation in dataset.annotations:
            if target_task == DatasetTask.INSTANCE_SEGMENTATION and not isinstance(
                annotation.geometry,
                MultiPolygon,
            ):
                validation.add(
                    "error",
                    "SEGMENTATION_REQUIRED",
                    "Segmentation merge requires polygon segmentation annotations",
                    annotation_id=annotation.id,
                )
                continue
            if target_task == DatasetTask.ORIENTED_DETECTION and not _can_merge_as_obb(
                annotation.geometry
            ):
                validation.add(
                    "error",
                    "OBB_QUADRILATERAL_REQUIRED",
                    "OBB merge requires bbox, OBB, or single quadrilateral polygon annotations",
                    annotation_id=annotation.id,
                )
                continue
            new_image_id = image_map[annotation.image_id]
            new_category_id = category_maps[dataset_index][annotation.category_id]
            signature = (
                new_image_id,
                new_category_id,
                annotation.geometry.model_dump_json(),
            )
            if signature in annotation_signatures:
                report_duplicate = str(annotation.id)
                validation.add(
                    "warning",
                    "DUPLICATE_ANNOTATION_SKIPPED",
                    f"Duplicate annotation skipped: {report_duplicate}",
                    annotation_id=annotation.id,
                )
                continue
            annotation_signatures.add(signature)
            merged_annotations.append(
                Annotation(
                    id=len(merged_annotations) + 1,
                    image_id=new_image_id,
                    category_id=new_category_id,
                    task_type=annotation.task_type,
                    geometry=annotation.geometry,
                    attributes=annotation.attributes,
                    source=annotation.source,
                )
            )

    if validation.has_errors:
        return None
    return Dataset(
        id="merged",
        name="merged",
        dataset_type=target_task,
        root=datasets[0].root,
        classes=merged_classes,
        images=merged_images,
        annotations=merged_annotations,
        splits={"train": [image.id for image in merged_images]},
        metadata={"source_count": len(datasets), "class_policy": class_policy},
        provenance={"operation": "merge"},
    )


def _unique_file_name(file_name: str, used_file_names: set[str]) -> str:
    candidate = file_name
    suffix = 1
    path = Path(file_name)
    while candidate in used_file_names:
        candidate = f"{path.stem}_{suffix}{path.suffix}"
        suffix += 1
    used_file_names.add(candidate)
    return candidate


def _task_from_cli(task: str) -> DatasetTask:
    if task == "segmentation":
        return DatasetTask.INSTANCE_SEGMENTATION
    if task == "obb":
        return DatasetTask.ORIENTED_DETECTION
    return DatasetTask.DETECTION


def _can_merge_as_obb(geometry) -> bool:
    if isinstance(geometry, OrientedBBox):
        return True
    if isinstance(geometry, RLEMask):
        return False
    if isinstance(geometry, MultiPolygon):
        return len(geometry.polygons) == 1 and len(geometry.polygons[0].points) == 4
    return True
