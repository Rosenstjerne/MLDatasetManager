from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def coco_detection_dataset(tmp_path: Path) -> Path:
    return _make_coco_dataset(tmp_path, segmentation=[])


@pytest.fixture()
def coco_segmentation_dataset(tmp_path: Path) -> Path:
    return _make_coco_dataset(tmp_path, segmentation=[[1, 1, 8, 1, 8, 8, 1, 8]])


@pytest.fixture()
def yolo_detection_dataset(tmp_path: Path) -> Path:
    return _make_yolo_dataset(tmp_path, "0 0.5 0.5 0.7 0.7\n", task="detect")


@pytest.fixture()
def yolo_segmentation_dataset(tmp_path: Path) -> Path:
    return _make_yolo_dataset(tmp_path, "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n", task="segment")


@pytest.fixture()
def yolo_obb_dataset(tmp_path: Path) -> Path:
    return _make_yolo_dataset(tmp_path, "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n", task="obb")


def _make_coco_dataset(tmp_path: Path, segmentation) -> Path:
    root = tmp_path / "coco" / "train"
    root.mkdir(parents=True)
    Image.new("RGB", (10, 10), color="white").save(root / "image1.jpg")
    annotation = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 7, 7],
                "area": 49,
                "segmentation": segmentation,
                "iscrowd": 0,
            }
        ],
    }
    (root / "_annotations.coco.json").write_text(json.dumps(annotation), encoding="utf-8")
    return root


def _make_yolo_dataset(tmp_path: Path, label: str, task: str) -> Path:
    root = tmp_path / "yolo"
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), color="white").save(images_dir / "image1.jpg")
    (labels_dir / "image1.txt").write_text(label, encoding="utf-8")
    (root / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/train",
                "nc: 1",
                "names:",
                "  0: object",
                f"task: {task}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root
