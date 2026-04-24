from typer.testing import CliRunner

from mldatasetmanager.cli.main import app


def test_cli_inspect(coco_detection_dataset):
    result = CliRunner().invoke(app, ["inspect", str(coco_detection_dataset), "--format", "coco"])

    assert result.exit_code == 0
    assert "Images: 1" in result.stdout


def test_cli_validate_fails_on_invalid_dataset(coco_detection_dataset):
    (coco_detection_dataset / "image1.jpg").unlink()

    result = CliRunner().invoke(app, ["validate", str(coco_detection_dataset), "--format", "coco"])

    assert result.exit_code == 1
    assert "1 errors" in result.stdout


def test_cli_convert(coco_detection_dataset, tmp_path):
    output = tmp_path / "yolo"

    result = CliRunner().invoke(
        app,
        [
            "convert",
            str(coco_detection_dataset),
            str(output),
            "--from",
            "coco",
            "--to",
            "yolo",
            "--task",
            "detection",
        ],
    )

    assert result.exit_code == 0
    assert (output / "data.yaml").exists()


def test_cli_convert_yolo_to_coco(yolo_detection_dataset, tmp_path):
    output = tmp_path / "coco"

    result = CliRunner().invoke(
        app,
        [
            "convert",
            str(yolo_detection_dataset),
            str(output),
            "--from",
            "yolo",
            "--to",
            "coco",
            "--task",
            "detection",
        ],
    )

    assert result.exit_code == 0
    assert (output / "_annotations.coco.json").exists()
