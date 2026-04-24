import json

from typer.testing import CliRunner

from mldatasetmanager.api import merge_datasets
from mldatasetmanager.cli.main import app


def test_merge_two_coco_datasets_to_coco(coco_detection_dataset, tmp_path):
    second = _copy_coco_dataset_with_name(coco_detection_dataset, tmp_path / "second", "image2.jpg")
    output = tmp_path / "merged-coco"

    report = merge_datasets(
        [coco_detection_dataset, second],
        output,
        source_formats=["coco", "coco"],
        target_format="coco",
        task="detection",
    )

    assert report.success
    data = json.loads((output / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 2
    assert len(data["categories"]) == 1


def test_merge_coco_and_yolo_to_yolo(coco_detection_dataset, yolo_detection_dataset, tmp_path):
    output = tmp_path / "merged-yolo"

    report = merge_datasets(
        [coco_detection_dataset, yolo_detection_dataset],
        output,
        source_formats=["coco", "yolo"],
        target_format="yolo",
        task="detection",
    )

    assert report.success
    assert len(list((output / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((output / "labels" / "train").glob("*.txt"))) == 2
    assert (output / "data.yaml").exists()


def test_merge_exact_class_policy_rejects_new_classes(
    coco_detection_dataset,
    yolo_detection_dataset,
    tmp_path,
):
    data_yaml = yolo_detection_dataset / "data.yaml"
    data_yaml.write_text(
        data_yaml.read_text(encoding="utf-8").replace("object", "different"),
        encoding="utf-8",
    )
    output = tmp_path / "merged-coco"

    report = merge_datasets(
        [coco_detection_dataset, yolo_detection_dataset],
        output,
        source_formats=["coco", "yolo"],
        target_format="coco",
        task="detection",
        options={"class_policy": "exact"},
    )

    assert not report.success
    assert any(item.code == "CLASS_MISSING_IN_MAIN" for item in report.validation.errors)
    assert not output.exists()


def test_merge_renames_colliding_image_filenames(
    coco_detection_dataset,
    yolo_detection_dataset,
    tmp_path,
):
    output = tmp_path / "merged-coco"

    report = merge_datasets(
        [coco_detection_dataset, yolo_detection_dataset],
        output,
        source_formats=["coco", "yolo"],
        target_format="coco",
        task="detection",
    )

    assert report.success
    data = json.loads((output / "_annotations.coco.json").read_text(encoding="utf-8"))
    file_names = [image["file_name"] for image in data["images"]]
    assert file_names == ["image1.jpg", "image1_1.jpg"]
    assert (output / "image1.jpg").exists()
    assert (output / "image1_1.jpg").exists()


def test_cli_merge_coco_and_yolo_to_coco(coco_detection_dataset, yolo_detection_dataset, tmp_path):
    output = tmp_path / "merged-coco"

    result = CliRunner().invoke(
        app,
        [
            "merge",
            str(output),
            str(coco_detection_dataset),
            str(yolo_detection_dataset),
            "--formats",
            "coco,yolo",
            "--to",
            "coco",
            "--task",
            "detection",
        ],
    )

    assert result.exit_code == 0
    assert (output / "_annotations.coco.json").exists()


def _copy_coco_dataset_with_name(source, target, image_name):
    target.mkdir(parents=True)
    (target / image_name).write_bytes((source / "image1.jpg").read_bytes())
    data = json.loads((source / "_annotations.coco.json").read_text(encoding="utf-8"))
    data["images"][0]["file_name"] = image_name
    (target / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    return target
