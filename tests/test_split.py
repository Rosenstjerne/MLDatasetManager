import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from mldatasetmanager.api import resplit_dataset
from mldatasetmanager.cli.main import app


def test_resplit_coco_to_yolo_writes_requested_counts(tmp_path):
    source = _make_multi_class_coco(tmp_path, image_count=10)
    output = tmp_path / "split-yolo"

    report = resplit_dataset(
        source,
        output,
        source_format="coco",
        target_format="yolo",
        task="detection",
        ratios=[60, 20, 20],
        seed=7,
    )

    assert report.success
    assert report.summary["splits"] == {"train": 6, "val": 2, "test": 2}
    assert len(list((output / "images" / "train").glob("*.jpg"))) == 6
    assert len(list((output / "images" / "val").glob("*.jpg"))) == 2
    assert len(list((output / "images" / "test").glob("*.jpg"))) == 2


def test_resplit_coco_to_coco_writes_split_annotation_files(tmp_path):
    source = _make_multi_class_coco(tmp_path, image_count=10)
    output = tmp_path / "split-coco"

    report = resplit_dataset(
        source,
        output,
        source_format="coco",
        target_format="coco",
        task="detection",
        ratios=[60, 20, 20],
        seed=7,
    )

    assert report.success
    assert _coco_image_count(output / "train") == 6
    assert _coco_image_count(output / "val") == 2
    assert _coco_image_count(output / "test") == 2


def test_resplit_with_class_stratification_balances_primary_classes(tmp_path):
    source = _make_multi_class_coco(tmp_path, image_count=10)
    output = tmp_path / "split-coco"

    report = resplit_dataset(
        source,
        output,
        source_format="coco",
        target_format="coco",
        task="detection",
        ratios=[60, 20, 20],
        seed=3,
        stratify="class",
    )

    assert report.success
    for split_name in ("train", "val", "test"):
        category_ids = _coco_annotation_category_ids(output / split_name)
        assert category_ids == {0, 1}


def test_resplit_refuses_existing_output_without_overwrite(tmp_path):
    source = _make_multi_class_coco(tmp_path, image_count=4)
    output = tmp_path / "split-yolo"
    output.mkdir()

    report = resplit_dataset(
        source,
        output,
        source_format="coco",
        target_format="yolo",
        task="detection",
        ratios=[50, 25, 25],
    )

    assert not report.success
    assert any(item.code == "OUTPUT_EXISTS" for item in report.validation.errors)


def test_cli_split_coco_to_yolo(tmp_path):
    source = _make_multi_class_coco(tmp_path, image_count=6)
    output = tmp_path / "split-yolo"

    result = CliRunner().invoke(
        app,
        [
            "split",
            str(source),
            str(output),
            "--format",
            "coco",
            "--to",
            "yolo",
            "--task",
            "detection",
            "--ratios",
            "50,25,25",
            "--seed",
            "11",
        ],
    )

    assert result.exit_code == 0
    assert (output / "data.yaml").exists()
    assert len(list((output / "images" / "train").glob("*.jpg"))) == 3


def _make_multi_class_coco(tmp_path: Path, image_count: int) -> Path:
    root = tmp_path / "multi-coco"
    root.mkdir()
    images = []
    annotations = []
    for index in range(image_count):
        file_name = f"image{index}.jpg"
        Image.new("RGB", (10, 10), color="white").save(root / file_name)
        images.append({"id": index, "file_name": file_name, "width": 10, "height": 10})
        annotations.append(
            {
                "id": index + 1,
                "image_id": index,
                "category_id": index % 2,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "segmentation": [],
                "iscrowd": 0,
            }
        )
    data = {
        "images": images,
        "categories": [
            {"id": 0, "name": "class-a", "supercategory": "none"},
            {"id": 1, "name": "class-b", "supercategory": "none"},
        ],
        "annotations": annotations,
    }
    (root / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def _coco_image_count(path: Path) -> int:
    data = json.loads((path / "_annotations.coco.json").read_text(encoding="utf-8"))
    return len(data["images"])


def _coco_annotation_category_ids(path: Path) -> set[int]:
    data = json.loads((path / "_annotations.coco.json").read_text(encoding="utf-8"))
    return {annotation["category_id"] for annotation in data["annotations"]}
