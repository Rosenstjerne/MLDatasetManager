import json

import pytest
from pydantic import ValidationError

from mldatasetmanager.adapters.yolo import YoloAdapter, try_read_yolo
from mldatasetmanager.api import convert_dataset, merge_datasets, resplit_dataset
from mldatasetmanager.core.models import (
    Annotation,
    Category,
    Dataset,
    DatasetTask,
    ImageAsset,
    OrientedBBox,
)
from mldatasetmanager.validation.validators import validate_dataset


def test_oriented_bbox_accepts_four_points():
    obb = OrientedBBox(points=[(1, 1), (8, 1), (8, 8), (1, 8)])

    assert len(obb.points) == 4


def test_oriented_bbox_rejects_non_four_point_shapes():
    with pytest.raises(ValidationError):
        OrientedBBox(points=[(1, 1), (8, 1), (8, 8)])


def test_validation_rejects_zero_area_obb(tmp_path):
    image_path = tmp_path / "image.jpg"
    from PIL import Image

    Image.new("RGB", (10, 10), color="white").save(image_path)
    dataset = Dataset(
        id="obb",
        name="obb",
        dataset_type=DatasetTask.ORIENTED_DETECTION,
        root=tmp_path,
        classes=[Category(id=0, name="object")],
        images=[ImageAsset(id=0, path=image_path, width=10, height=10)],
        annotations=[
            Annotation(
                id=1,
                image_id=0,
                category_id=0,
                task_type=DatasetTask.ORIENTED_DETECTION,
                geometry=OrientedBBox(points=[(1, 1), (2, 2), (3, 3), (4, 4)]),
            )
        ],
    )

    report = validate_dataset(dataset)

    assert report.has_errors
    assert any(item.code == "OBB_AREA_INVALID" for item in report.errors)


def test_yolo_reader_loads_obb_dataset(yolo_obb_dataset):
    dataset = YoloAdapter().read(yolo_obb_dataset)

    assert dataset.dataset_type == DatasetTask.ORIENTED_DETECTION
    assert isinstance(dataset.annotations[0].geometry, OrientedBBox)


def test_yolo_writer_preserves_obb_rows(yolo_obb_dataset, tmp_path):
    dataset = YoloAdapter().read(yolo_obb_dataset)
    output = tmp_path / "out"

    files_written = YoloAdapter().write(dataset, output, {"task": "oriented-detection"})

    assert files_written == 3
    row = (output / "labels" / "train" / "image1.txt").read_text(encoding="utf-8").strip()
    assert row == "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8"
    assert "task: obb" in (output / "data.yaml").read_text(encoding="utf-8")


def test_yolo_reader_rejects_malformed_obb_rows(yolo_obb_dataset):
    label_path = yolo_obb_dataset / "labels" / "train" / "image1.txt"
    label_path.write_text("0 0.1 0.1 0.8 0.1\n", encoding="utf-8")

    dataset, report = try_read_yolo(yolo_obb_dataset)

    assert dataset is None
    assert any(item.code == "YOLO_READ_FAILED" for item in report.errors)


def test_yolo_obb_to_coco_writes_polygon_segmentation(yolo_obb_dataset, tmp_path):
    output = tmp_path / "coco"

    report = convert_dataset(yolo_obb_dataset, output, "yolo", "coco", "obb")

    assert report.success
    data = json.loads((output / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert data["annotations"][0]["segmentation"] == [[1.0, 1.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0]]


def test_coco_quadrilateral_segmentation_to_yolo_obb(coco_segmentation_dataset, tmp_path):
    output = tmp_path / "yolo-obb"

    report = convert_dataset(coco_segmentation_dataset, output, "coco", "yolo", "obb")

    assert report.success
    row = (output / "labels" / "train" / "image1.txt").read_text(encoding="utf-8").strip()
    assert row == "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8"


def test_coco_detection_to_yolo_obb(coco_detection_dataset, tmp_path):
    output = tmp_path / "yolo-obb"

    report = convert_dataset(coco_detection_dataset, output, "coco", "yolo", "obb")

    assert report.success
    row = (output / "labels" / "train" / "image1.txt").read_text(encoding="utf-8").strip()
    assert row == "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8"


def test_yolo_obb_to_detection_requires_lossy_opt_in(yolo_obb_dataset, tmp_path):
    output = tmp_path / "detect"

    report = convert_dataset(yolo_obb_dataset, output, "yolo", "yolo", "detection")

    assert not report.success
    assert any(item.code == "LOSSY_CONVERSION_REQUIRES_OPT_IN" for item in report.validation.errors)
    assert not output.exists()


def test_yolo_obb_to_detection_succeeds_with_lossy_opt_in(yolo_obb_dataset, tmp_path):
    output = tmp_path / "detect"

    report = convert_dataset(
        yolo_obb_dataset,
        output,
        "yolo",
        "yolo",
        "detection",
        options={"allow_lossy": True},
    )

    assert report.success
    row = (output / "labels" / "train" / "image1.txt").read_text(encoding="utf-8").strip()
    assert row == "0 0.45 0.45 0.7 0.7"


def test_merge_two_yolo_obb_datasets(yolo_obb_dataset, tmp_path):
    second = tmp_path / "second"
    convert_dataset(yolo_obb_dataset, second, "yolo", "yolo", "obb")
    output = tmp_path / "merged"

    report = merge_datasets(
        [yolo_obb_dataset, second],
        output,
        source_formats=["yolo", "yolo"],
        target_format="yolo",
        task="obb",
    )

    assert report.success
    assert (output / "data.yaml").exists()
    assert "task: obb" in (output / "data.yaml").read_text(encoding="utf-8")


def test_split_yolo_obb_dataset(yolo_obb_dataset, tmp_path):
    output = tmp_path / "split"

    report = resplit_dataset(
        yolo_obb_dataset,
        output,
        source_format="yolo",
        target_format="yolo",
        task="obb",
        ratios=[1, 0, 0],
    )

    assert report.success
    assert (output / "labels" / "train" / "image1.txt").exists()
