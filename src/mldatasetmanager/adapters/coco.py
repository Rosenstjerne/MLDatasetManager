from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mldatasetmanager.adapters.base import AdapterCapabilities, DetectionResult
from mldatasetmanager.core.models import (
    Annotation,
    AxisAlignedBBox,
    Category,
    Dataset,
    DatasetTask,
    ImageAsset,
    MultiPolygon,
    OrientedBBox,
    Polygon,
    RLEMask,
)
from mldatasetmanager.reports import ValidationReport


class CocoAdapter:
    format_name = "coco"
    capabilities = AdapterCapabilities(
        tasks={"detection", "instance-segmentation"},
        geometries={"bbox", "polygon", "multipolygon", "rle"},
    )

    def detect(self, path: Path) -> DetectionResult:
        annotation_path = self._annotation_path(path)
        if annotation_path.exists():
            return DetectionResult(True, 0.95, f"found {annotation_path.name}")
        return DetectionResult(False, 0.0, "missing COCO annotation JSON")

    def validate_structure(self, path: Path) -> ValidationReport:
        report = ValidationReport(dataset_path=str(path))
        annotation_path = self._annotation_path(path)
        if not path.exists():
            report.add(
                "error", "DATASET_PATH_MISSING", "Dataset path does not exist", file_path=str(path)
            )
            return report
        if not annotation_path.exists():
            report.add(
                "error",
                "COCO_ANNOTATION_MISSING",
                "COCO annotation file was not found",
                file_path=str(annotation_path),
            )
            return report
        try:
            data = self._load_json(annotation_path)
        except json.JSONDecodeError as exc:
            report.add(
                "error",
                "COCO_JSON_INVALID",
                f"Annotation JSON is invalid: {exc}",
                file_path=str(annotation_path),
            )
            return report
        for key in ("images", "annotations", "categories"):
            if key not in data or not isinstance(data[key], list):
                report.add(
                    "error",
                    "COCO_REQUIRED_FIELD_MISSING",
                    f"COCO field '{key}' must exist and be a list",
                    file_path=str(annotation_path),
                )
        report.summary = {
            "images": len(data.get("images", [])) if isinstance(data.get("images"), list) else 0,
            "annotations": len(data.get("annotations", []))
            if isinstance(data.get("annotations"), list)
            else 0,
            "categories": len(data.get("categories", []))
            if isinstance(data.get("categories"), list)
            else 0,
        }
        return report

    def read(self, path: Path, options: dict | None = None) -> Dataset:
        annotation_path = self._annotation_path(path)
        data = self._load_json(annotation_path)
        categories = [
            Category(
                id=int(item["id"]),
                name=str(item["name"]),
                supercategory=item.get("supercategory"),
                metadata={
                    k: v for k, v in item.items() if k not in {"id", "name", "supercategory"}
                },
            )
            for item in data.get("categories", [])
        ]
        images = [
            ImageAsset(
                id=item["id"],
                path=path / item["file_name"],
                width=int(item["width"]),
                height=int(item["height"]),
                split=path.name if path.name in {"train", "valid", "val", "test"} else None,
                metadata={
                    k: v for k, v in item.items() if k not in {"id", "file_name", "width", "height"}
                },
            )
            for item in data.get("images", [])
        ]
        annotations = [self._read_annotation(item) for item in data.get("annotations", [])]
        dataset_type = (
            DatasetTask.INSTANCE_SEGMENTATION
            if any(item.task_type == DatasetTask.INSTANCE_SEGMENTATION for item in annotations)
            else DatasetTask.DETECTION
        )
        return Dataset(
            id=path.name,
            name=path.parent.name if path.name in {"train", "valid", "val", "test"} else path.name,
            dataset_type=dataset_type,
            root=path,
            classes=categories,
            images=images,
            annotations=annotations,
            splits={path.name: [image.id for image in images]}
            if path.name in {"train", "valid", "val", "test"}
            else {},
            metadata={
                k: v for k, v in data.items() if k not in {"images", "annotations", "categories"}
            },
            provenance={"format": "coco", "annotation_file": str(annotation_path)},
        )

    def write(self, dataset: Dataset, path: Path, options: dict | None = None) -> int:
        options = options or {}
        if options.get("split_output"):
            return self._write_split_dataset(dataset, path)
        return self._write_single_dataset(dataset, path)

    def _write_split_dataset(self, dataset: Dataset, path: Path) -> int:
        files_written = 0
        split_names = [name for name, image_ids in dataset.splits.items() if image_ids]
        if not split_names:
            return self._write_single_dataset(dataset, path)
        for split_name in split_names:
            split_path = path / split_name
            split_image_ids = set(dataset.splits[split_name])
            split_dataset = Dataset(
                id=f"{dataset.id}-{split_name}",
                name=dataset.name,
                dataset_type=dataset.dataset_type,
                root=dataset.root,
                classes=dataset.classes,
                images=[image for image in dataset.images if image.id in split_image_ids],
                annotations=[
                    annotation
                    for annotation in dataset.annotations
                    if annotation.image_id in split_image_ids
                ],
                splits={split_name: list(split_image_ids)},
                metadata=dataset.metadata,
                provenance=dataset.provenance,
            )
            files_written += self._write_single_dataset(split_dataset, split_path)
        return files_written

    def _write_single_dataset(self, dataset: Dataset, path: Path) -> int:
        path.mkdir(parents=True, exist_ok=True)
        image_id_map = {image.id: index for index, image in enumerate(dataset.images)}
        annotation_items = []
        files_written = 0

        for image in dataset.images:
            shutil.copy2(image.path, path / _target_file_name(image))
            files_written += 1

        for index, annotation in enumerate(dataset.annotations, start=1):
            image = dataset.image_by_id[annotation.image_id]
            annotation_items.append(
                self._write_annotation(annotation, image_id_map[annotation.image_id], index, image)
            )

        coco = {
            "info": {
                "description": dataset.name,
                "version": "mldatasetmanager",
            },
            "licenses": [],
            "categories": [
                {
                    "id": category.id,
                    "name": category.name,
                    "supercategory": category.supercategory or "none",
                }
                for category in dataset.classes
            ],
            "images": [
                {
                    "id": image_id_map[image.id],
                    "file_name": _target_file_name(image),
                    "width": image.width,
                    "height": image.height,
                }
                for image in dataset.images
            ],
            "annotations": annotation_items,
        }
        (path / "_annotations.coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
        return files_written + 1

    def _write_annotation(
        self,
        annotation: Annotation,
        image_id: int,
        annotation_id: int,
        image: ImageAsset,
    ) -> dict[str, Any]:
        if isinstance(annotation.geometry, AxisAlignedBBox):
            bbox = annotation.geometry
            segmentation: list[list[float]] = []
            area = bbox.width * bbox.height
        elif isinstance(annotation.geometry, MultiPolygon):
            polygons = annotation.geometry.polygons
            xs = [x for polygon in polygons for x, _ in polygon.points]
            ys = [y for polygon in polygons for _, y in polygon.points]
            bbox = AxisAlignedBBox(
                x_min=max(0.0, min(xs)),
                y_min=max(0.0, min(ys)),
                x_max=min(float(image.width), max(xs)),
                y_max=min(float(image.height), max(ys)),
            )
            segmentation = [
                [coordinate for point in polygon.points for coordinate in point]
                for polygon in polygons
            ]
            area = sum(abs(_polygon_area(polygon.points)) for polygon in polygons)
        elif isinstance(annotation.geometry, OrientedBBox):
            xs = [x for x, _ in annotation.geometry.points]
            ys = [y for _, y in annotation.geometry.points]
            bbox = AxisAlignedBBox(
                x_min=max(0.0, min(xs)),
                y_min=max(0.0, min(ys)),
                x_max=min(float(image.width), max(xs)),
                y_max=min(float(image.height), max(ys)),
            )
            segmentation = [
                [coordinate for point in annotation.geometry.points for coordinate in point]
            ]
            area = abs(_polygon_area(annotation.geometry.points))
        else:
            raise ValueError(f"Unsupported COCO export geometry: {annotation.geometry.kind}")

        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": annotation.category_id,
            "bbox": [
                _clean_float(value) for value in [bbox.x_min, bbox.y_min, bbox.width, bbox.height]
            ],
            "area": _clean_float(area),
            "segmentation": [
                [_clean_float(value) for value in polygon] for polygon in segmentation
            ],
            "iscrowd": 0,
        }

    def _read_annotation(self, item: dict[str, Any]) -> Annotation:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"COCO annotation {item.get('id')} is missing a valid bbox")
        x, y, width, height = (float(value) for value in bbox)
        segmentation = item.get("segmentation")
        geometry = self._read_segmentation(segmentation)
        task_type = (
            DatasetTask.INSTANCE_SEGMENTATION if geometry is not None else DatasetTask.DETECTION
        )
        if geometry is None:
            geometry = AxisAlignedBBox(x_min=x, y_min=y, x_max=x + width, y_max=y + height)
        return Annotation(
            id=item["id"],
            image_id=item["image_id"],
            category_id=int(item["category_id"]),
            task_type=task_type,
            geometry=geometry,
            attributes={
                "bbox": [x, y, width, height],
                "area": item.get("area"),
                "iscrowd": item.get("iscrowd", 0),
            },
            source={"format": "coco", "raw": item},
        )

    def _read_segmentation(self, segmentation: Any) -> MultiPolygon | RLEMask | None:
        if not segmentation:
            return None
        if isinstance(segmentation, dict):
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            if isinstance(size, list) and len(size) == 2 and counts is not None:
                return RLEMask(size=(int(size[0]), int(size[1])), counts=counts)
            return None
        if isinstance(segmentation, list):
            polygons = []
            for raw_polygon in segmentation:
                if (
                    not isinstance(raw_polygon, list)
                    or len(raw_polygon) < 6
                    or len(raw_polygon) % 2 != 0
                ):
                    raise ValueError("COCO polygon segmentation must contain x/y pairs")
                points = [
                    (float(raw_polygon[index]), float(raw_polygon[index + 1]))
                    for index in range(0, len(raw_polygon), 2)
                ]
                polygons.append(Polygon(points=points))
            if polygons:
                return MultiPolygon(polygons=polygons)
        return None

    def _annotation_path(self, path: Path) -> Path:
        if path.is_file():
            return path
        candidates = [
            path / "_annotations.coco.json",
            path / "annotations.coco.json",
            path / "annotations" / "instances.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def try_read_coco(path: Path) -> tuple[Dataset | None, ValidationReport]:
    adapter = CocoAdapter()
    report = adapter.validate_structure(path)
    if report.has_errors:
        return None, report
    try:
        return adapter.read(path), report
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        report.add("error", "COCO_READ_FAILED", f"Failed to read COCO dataset: {exc}")
        return None, report


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2


def _clean_float(value: float) -> float:
    return round(float(value), 6)


def _target_file_name(image: ImageAsset) -> str:
    return str(image.metadata.get("target_file_name") or image.path.name)
