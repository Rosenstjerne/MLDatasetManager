from __future__ import annotations

import ast
import shutil
from pathlib import Path

from PIL import Image, ImageOps
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
    Polygon,
)
from mldatasetmanager.reports import ValidationReport


class YoloAdapter:
    format_name = "yolo"
    capabilities = AdapterCapabilities(
        tasks={"detection", "instance-segmentation"},
        geometries={"bbox", "polygon"},
        coordinate_system="normalized",
    )

    def detect(self, path: Path) -> DetectionResult:
        if (path / "data.yaml").exists():
            return DetectionResult(True, 0.8, "found data.yaml")
        return DetectionResult(False, 0.0, "missing data.yaml")

    def validate_structure(self, path: Path) -> ValidationReport:
        report = ValidationReport(dataset_path=str(path))
        if not path.exists():
            report.add(
                "error", "DATASET_PATH_MISSING", "Dataset path does not exist", file_path=str(path)
            )
            return report
        if not (path / "data.yaml").exists():
            report.add(
                "error",
                "YOLO_DATA_YAML_MISSING",
                "YOLO data.yaml file was not found",
                file_path=str(path / "data.yaml"),
            )
        if not (path / "images").exists():
            report.add(
                "error",
                "YOLO_IMAGES_DIR_MISSING",
                "YOLO images directory was not found",
                file_path=str(path / "images"),
            )
        return report

    def read(self, path: Path, options: dict | None = None) -> Dataset:
        options = options or {}
        data = _read_yolo_data_yaml(path / "data.yaml")
        task = _task_from_options_or_yaml(options.get("task"), data.get("task"))
        categories = [
            Category(id=class_id, name=name, supercategory="none")
            for class_id, name in sorted(data["names"].items())
        ]
        split_paths = _split_paths(path, data)
        images: list[ImageAsset] = []
        annotations: list[Annotation] = []
        image_id = 0
        annotation_id = 1

        for split_name, images_dir in split_paths.items():
            labels_dir = _labels_dir_for_images_dir(path, images_dir)
            for image_path in _iter_images(images_dir):
                with Image.open(image_path) as opened:
                    width, height = ImageOps.exif_transpose(opened).size
                images.append(
                    ImageAsset(
                        id=image_id,
                        path=image_path,
                        width=width,
                        height=height,
                        split=split_name,
                    )
                )
                label_path = labels_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    for line_number, line in enumerate(
                        label_path.read_text(encoding="utf-8").splitlines(),
                        start=1,
                    ):
                        if not line.strip():
                            continue
                        annotation = _read_yolo_label_row(
                            line,
                            image_id=image_id,
                            annotation_id=annotation_id,
                            image_width=width,
                            image_height=height,
                            task=task,
                            category_ids={category.id for category in categories},
                            label_path=label_path,
                            line_number=line_number,
                        )
                        annotations.append(annotation)
                        annotation_id += 1
                image_id += 1

        return Dataset(
            id=path.name,
            name=path.name,
            dataset_type=task,
            root=path,
            classes=categories,
            images=images,
            annotations=annotations,
            splits={
                split_name: [image.id for image in images if image.split == split_name]
                for split_name in split_paths
            },
            metadata={"data_yaml": data},
            provenance={"format": "yolo", "data_yaml": str(path / "data.yaml")},
        )

    def write(self, dataset: Dataset, path: Path, options: dict | None = None) -> int:
        options = options or {}
        task = DatasetTask(options.get("task", dataset.dataset_type.value))
        if task not in {DatasetTask.DETECTION, DatasetTask.INSTANCE_SEGMENTATION}:
            raise ValueError(f"Unsupported YOLO task: {task}")

        images_dir = path / "images" / "train"
        labels_dir = path / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        category_map = {category.id: index for index, category in enumerate(dataset.classes)}
        annotations_by_image = {image.id: [] for image in dataset.images}
        for annotation in dataset.annotations:
            annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

        files_written = 0
        for image in dataset.images:
            target_image = images_dir / image.path.name
            shutil.copy2(image.path, target_image)
            files_written += 1

            label_path = labels_dir / f"{image.path.stem}.txt"
            rows = []
            for annotation in annotations_by_image.get(image.id, []):
                class_id = category_map[annotation.category_id]
                if task == DatasetTask.DETECTION:
                    rows.append(
                        self._format_detection_row(class_id, annotation, image.width, image.height)
                    )
                else:
                    rows.extend(
                        self._format_segmentation_rows(
                            class_id, annotation, image.width, image.height
                        )
                    )
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            files_written += 1

        self._write_data_yaml(path, dataset, task)
        return files_written + 1

    def _format_detection_row(
        self,
        class_id: int,
        annotation,
        image_width: int,
        image_height: int,
    ) -> str:
        bbox = self._bbox_from_annotation(annotation)
        x_center = (bbox.x_min + bbox.width / 2) / image_width
        y_center = (bbox.y_min + bbox.height / 2) / image_height
        width = bbox.width / image_width
        height = bbox.height / image_height
        return self._format_row([class_id, x_center, y_center, width, height])

    def _format_segmentation_rows(
        self,
        class_id: int,
        annotation,
        image_width: int,
        image_height: int,
    ) -> list[str]:
        geometry = annotation.geometry
        if not isinstance(geometry, MultiPolygon):
            return []
        rows = []
        for polygon in geometry.polygons:
            values: list[int | float] = [class_id]
            for x, y in polygon.points:
                values.extend([x / image_width, y / image_height])
            rows.append(self._format_row(values))
        return rows

    def _bbox_from_annotation(self, annotation) -> AxisAlignedBBox:
        if isinstance(annotation.geometry, AxisAlignedBBox):
            return annotation.geometry
        bbox = annotation.attributes.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, width, height = (float(value) for value in bbox)
            return AxisAlignedBBox(x_min=x, y_min=y, x_max=x + width, y_max=y + height)
        raise ValueError(f"Annotation {annotation.id} does not have a bbox")

    def _format_row(self, values: list[int | float]) -> str:
        formatted = []
        for value in values:
            if isinstance(value, int):
                formatted.append(str(value))
            else:
                formatted.append(f"{value:.6f}".rstrip("0").rstrip("."))
        return " ".join(formatted)

    def _write_data_yaml(self, path: Path, dataset: Dataset, task: DatasetTask) -> None:
        names = [category.name for category in dataset.classes]
        lines = [
            "path: .",
            "train: images/train",
            "val: images/train",
            f"nc: {len(names)}",
            "names:",
        ]
        lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
        lines.append(
            f"task: {'segment' if task == DatasetTask.INSTANCE_SEGMENTATION else 'detect'}"
        )
        (path / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_read_yolo(
    path: Path, options: dict | None = None
) -> tuple[Dataset | None, ValidationReport]:
    adapter = YoloAdapter()
    report = adapter.validate_structure(path)
    if report.has_errors:
        return None, report
    try:
        return adapter.read(path, options or {}), report
    except (KeyError, TypeError, ValueError, OSError, ValidationError) as exc:
        report.add("error", "YOLO_READ_FAILED", f"Failed to read YOLO dataset: {exc}")
        return None, report


def _read_yolo_data_yaml(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8").splitlines()
    result: dict = {"names": {}}
    index = 0
    while index < len(raw):
        line = raw[index].strip()
        index += 1
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "names":
            if value:
                result["names"] = _parse_inline_names(value)
            else:
                names: dict[int, str] = {}
                while index < len(raw) and raw[index].startswith((" ", "\t")):
                    nested = raw[index].strip()
                    index += 1
                    if not nested or ":" not in nested:
                        continue
                    raw_id, raw_name = nested.split(":", 1)
                    names[int(raw_id.strip())] = raw_name.strip().strip("\"'")
                result["names"] = names
        elif key in {"path", "train", "val", "valid", "test", "task"}:
            result[key] = value.strip("\"'")
    if not result["names"]:
        raise ValueError("YOLO data.yaml must define class names")
    return result


def _parse_inline_names(value: str) -> dict[int, str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return {index: str(name) for index, name in enumerate(parsed)}
    if isinstance(parsed, dict):
        return {int(class_id): str(name) for class_id, name in parsed.items()}
    raise ValueError("YOLO names must be a list or mapping")


def _task_from_options_or_yaml(option_task: str | None, yaml_task: str | None) -> DatasetTask:
    task = (option_task or yaml_task or "detect").lower()
    if task in {"detection", "detect"}:
        return DatasetTask.DETECTION
    if task in {"segmentation", "segment", "instance-segmentation"}:
        return DatasetTask.INSTANCE_SEGMENTATION
    raise ValueError(f"Unsupported YOLO task: {task}")


def _split_paths(root: Path, data: dict) -> dict[str, Path]:
    base = root / data.get("path", ".")
    split_paths: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for split_name, yaml_key in {"train": "train", "val": "val", "test": "test"}.items():
        value = data.get(yaml_key)
        if value is None and yaml_key == "val":
            value = data.get("valid")
        if value is None:
            continue
        candidate = base / value
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            split_paths[split_name] = candidate
    if not split_paths:
        fallback = root / "images" / "train"
        if fallback.exists():
            split_paths["train"] = fallback
    if not split_paths:
        raise ValueError("YOLO dataset does not contain any readable image split directories")
    return split_paths


def _labels_dir_for_images_dir(root: Path, images_dir: Path) -> Path:
    parts = list(images_dir.relative_to(root).parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
        return root.joinpath(*parts)
    return root / "labels" / images_dir.name


def _iter_images(path: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions
    )


def _read_yolo_label_row(
    line: str,
    image_id: int,
    annotation_id: int,
    image_width: int,
    image_height: int,
    task: DatasetTask,
    category_ids: set[int],
    label_path: Path,
    line_number: int,
) -> Annotation:
    try:
        values = [float(item) for item in line.split()]
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number} contains a non-numeric label value") from exc
    if len(values) < 5:
        raise ValueError(f"{label_path}:{line_number} must contain at least 5 values")
    class_id = int(values[0])
    if values[0] != class_id:
        raise ValueError(f"{label_path}:{line_number} class id must be an integer")
    if class_id not in category_ids:
        raise ValueError(f"{label_path}:{line_number} references unknown class id {class_id}")
    if task == DatasetTask.INSTANCE_SEGMENTATION or len(values) > 5:
        if len(values) < 7 or len(values[1:]) % 2 != 0:
            raise ValueError(f"{label_path}:{line_number} segmentation row must contain x/y pairs")
        points = [
            (values[index] * image_width, values[index + 1] * image_height)
            for index in range(1, len(values), 2)
        ]
        geometry = MultiPolygon(polygons=[Polygon(points=points)])
        bbox = _bbox_from_points(points)
        return Annotation(
            id=annotation_id,
            image_id=image_id,
            category_id=class_id,
            task_type=DatasetTask.INSTANCE_SEGMENTATION,
            geometry=geometry,
            attributes={"bbox": [bbox.x_min, bbox.y_min, bbox.width, bbox.height]},
            source={"format": "yolo", "label_path": str(label_path), "line_number": line_number},
        )

    x_center, y_center, width, height = values[1:5]
    bbox = AxisAlignedBBox(
        x_min=(x_center - width / 2) * image_width,
        y_min=(y_center - height / 2) * image_height,
        x_max=(x_center + width / 2) * image_width,
        y_max=(y_center + height / 2) * image_height,
    )
    return Annotation(
        id=annotation_id,
        image_id=image_id,
        category_id=class_id,
        task_type=DatasetTask.DETECTION,
        geometry=bbox,
        attributes={"bbox": [bbox.x_min, bbox.y_min, bbox.width, bbox.height]},
        source={"format": "yolo", "label_path": str(label_path), "line_number": line_number},
    )


def _bbox_from_points(points: list[tuple[float, float]]) -> AxisAlignedBBox:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return AxisAlignedBBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))
