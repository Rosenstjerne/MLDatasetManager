from __future__ import annotations

import math

from PIL import Image, ImageOps, UnidentifiedImageError

from mldatasetmanager.core.models import (
    AxisAlignedBBox,
    Dataset,
    MultiPolygon,
    OrientedBBox,
    RLEMask,
)
from mldatasetmanager.reports import ValidationReport


def validate_dataset(dataset: Dataset) -> ValidationReport:
    report = ValidationReport(
        dataset_path=str(dataset.root),
        summary={
            "images": len(dataset.images),
            "annotations": len(dataset.annotations),
            "categories": len(dataset.classes),
        },
    )
    _validate_classes(dataset, report)
    _validate_images(dataset, report)
    _validate_annotations(dataset, report)
    return report


def _validate_classes(dataset: Dataset, report: ValidationReport) -> None:
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for category in dataset.classes:
        if category.id in seen_ids:
            report.add(
                "error",
                "DUPLICATE_CATEGORY_ID",
                f"Duplicate category id: {category.id}",
                category_id=category.id,
            )
        seen_ids.add(category.id)
        normalized_name = category.name.strip().lower()
        if not normalized_name:
            report.add(
                "error",
                "CATEGORY_NAME_EMPTY",
                "Category name must not be empty",
                category_id=category.id,
            )
        if normalized_name in seen_names:
            report.add(
                "warning",
                "DUPLICATE_CATEGORY_NAME",
                f"Duplicate category name: {category.name}",
                category_id=category.id,
            )
        seen_names.add(normalized_name)


def _validate_images(dataset: Dataset, report: ValidationReport) -> None:
    seen_ids: set[int | str] = set()
    for image in dataset.images:
        if image.id in seen_ids:
            report.add(
                "error",
                "DUPLICATE_IMAGE_ID",
                f"Duplicate image id: {image.id}",
                image_id=image.id,
                file_path=str(image.path),
            )
        seen_ids.add(image.id)
        if not image.path.exists():
            report.add(
                "error",
                "MISSING_IMAGE",
                "Image file does not exist",
                image_id=image.id,
                file_path=str(image.path),
            )
            continue
        try:
            with Image.open(image.path) as opened:
                actual_width, actual_height = opened.size
                oriented_width, oriented_height = ImageOps.exif_transpose(opened).size
        except (OSError, UnidentifiedImageError) as exc:
            report.add(
                "error",
                "IMAGE_UNREADABLE",
                f"Image file could not be read: {exc}",
                image_id=image.id,
                file_path=str(image.path),
            )
            continue
        if actual_width == image.width and actual_height == image.height:
            continue
        if oriented_width == image.width and oriented_height == image.height:
            report.add(
                "info",
                "IMAGE_EXIF_ORIENTATION_MATCH",
                "Image dimensions match annotation metadata after applying EXIF orientation",
                image_id=image.id,
                file_path=str(image.path),
            )
            continue
        else:
            report.add(
                "warning",
                "IMAGE_DIMENSION_MISMATCH",
                "Image dimensions differ from annotation metadata",
                image_id=image.id,
                file_path=str(image.path),
            )


def _validate_annotations(dataset: Dataset, report: ValidationReport) -> None:
    images = dataset.image_by_id
    categories = dataset.category_by_id
    seen_ids: set[int | str] = set()
    for annotation in dataset.annotations:
        if annotation.id in seen_ids:
            report.add(
                "error",
                "DUPLICATE_ANNOTATION_ID",
                f"Duplicate annotation id: {annotation.id}",
                annotation_id=annotation.id,
            )
        seen_ids.add(annotation.id)
        image = images.get(annotation.image_id)
        if image is None:
            report.add(
                "error",
                "UNKNOWN_IMAGE_REFERENCE",
                "Annotation references an unknown image",
                image_id=annotation.image_id,
                annotation_id=annotation.id,
            )
            continue
        if annotation.category_id not in categories:
            report.add(
                "error",
                "UNKNOWN_CATEGORY_REFERENCE",
                "Annotation references an unknown category",
                annotation_id=annotation.id,
                category_id=annotation.category_id,
            )
        geometry = annotation.geometry
        if isinstance(geometry, AxisAlignedBBox):
            _validate_bbox(geometry, image.width, image.height, report, annotation.id)
        elif isinstance(geometry, OrientedBBox):
            for x, y in geometry.points:
                _validate_point(x, y, image.width, image.height, report, annotation.id, "OBB")
            if _polygon_area(geometry.points) <= 0:
                report.add(
                    "error",
                    "OBB_AREA_INVALID",
                    "Oriented bbox must have positive polygon area",
                    annotation_id=annotation.id,
                )
        elif isinstance(geometry, MultiPolygon):
            for polygon in geometry.polygons:
                for x, y in polygon.points:
                    _validate_point(
                        x,
                        y,
                        image.width,
                        image.height,
                        report,
                        annotation.id,
                        "POLYGON",
                    )
        elif isinstance(geometry, RLEMask):
            if geometry.size[0] <= 0 or geometry.size[1] <= 0:
                report.add(
                    "error",
                    "RLE_SIZE_INVALID",
                    "RLE mask size must contain positive dimensions",
                    annotation_id=annotation.id,
                )


def _validate_bbox(
    bbox: AxisAlignedBBox,
    image_width: int,
    image_height: int,
    report: ValidationReport,
    annotation_id: int | str,
) -> None:
    for value in (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max):
        if not math.isfinite(value):
            report.add(
                "error",
                "BBOX_COORDINATE_INVALID",
                "BBox coordinate must be finite",
                annotation_id=annotation_id,
            )
            return
    if bbox.x_min < 0 or bbox.y_min < 0 or bbox.x_max > image_width or bbox.y_max > image_height:
        report.add(
            "error",
            "BBOX_OUT_OF_BOUNDS",
            "BBox coordinates must be inside image boundaries",
            annotation_id=annotation_id,
        )


def _validate_point(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    report: ValidationReport,
    annotation_id: int | str,
    prefix: str,
) -> None:
    if not math.isfinite(x) or not math.isfinite(y):
        report.add(
            "error",
            f"{prefix}_COORDINATE_INVALID",
            f"{prefix.title()} coordinate must be finite",
            annotation_id=annotation_id,
        )
        return
    if x < 0 or y < 0 or x > image_width or y > image_height:
        report.add(
            "error",
            f"{prefix}_OUT_OF_BOUNDS",
            f"{prefix.title()} coordinates must be inside image boundaries",
            annotation_id=annotation_id,
        )


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area / 2)
