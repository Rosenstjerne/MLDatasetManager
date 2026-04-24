from mldatasetmanager.api import convert_dataset


def test_detection_conversion_writes_yolo_dataset(coco_detection_dataset, tmp_path):
    output = tmp_path / "yolo"

    report = convert_dataset(coco_detection_dataset, output, "coco", "yolo", "detection")

    assert report.success
    assert (output / "data.yaml").exists()
    assert (output / "images" / "train" / "image1.jpg").exists()
    assert (output / "labels" / "train" / "image1.txt").read_text(encoding="utf-8").strip()


def test_segmentation_conversion_writes_yolo_polygon_rows(coco_segmentation_dataset, tmp_path):
    output = tmp_path / "yolo"

    report = convert_dataset(coco_segmentation_dataset, output, "coco", "yolo", "segmentation")

    assert report.success
    row = (output / "labels" / "train" / "image1.txt").read_text(encoding="utf-8").strip()
    assert len(row.split()) == 9


def test_invalid_dataset_does_not_write_output(coco_detection_dataset, tmp_path):
    (coco_detection_dataset / "image1.jpg").unlink()
    output = tmp_path / "yolo"

    report = convert_dataset(coco_detection_dataset, output, "coco", "yolo", "detection")

    assert not report.success
    assert not output.exists()


def test_detection_conversion_allows_rle_segmentation(coco_detection_dataset, tmp_path):
    import json

    annotation_path = coco_detection_dataset / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"][0]["segmentation"] = {"size": [10, 10], "counts": [100]}
    annotation_path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "yolo"

    report = convert_dataset(coco_detection_dataset, output, "coco", "yolo", "detection")

    assert report.success
    assert output.exists()


def test_segmentation_conversion_rejects_rle_segmentation(coco_detection_dataset, tmp_path):
    import json

    annotation_path = coco_detection_dataset / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"][0]["segmentation"] = {"size": [10, 10], "counts": [100]}
    annotation_path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "yolo"

    report = convert_dataset(coco_detection_dataset, output, "coco", "yolo", "segmentation")

    assert not report.success
    assert any(item.code == "RLE_SEGMENTATION_UNSUPPORTED" for item in report.validation.errors)
    assert not output.exists()


def test_yolo_detection_conversion_writes_coco_dataset(yolo_detection_dataset, tmp_path):
    import json

    output = tmp_path / "coco"

    report = convert_dataset(yolo_detection_dataset, output, "yolo", "coco", "detection")

    assert report.success
    annotation_path = output / "_annotations.coco.json"
    assert annotation_path.exists()
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert data["images"][0]["file_name"] == "image1.jpg"
    assert data["annotations"][0]["bbox"] == [1.5, 1.5, 7.0, 7.0]
    assert data["annotations"][0]["segmentation"] == []


def test_yolo_segmentation_conversion_writes_coco_polygons(yolo_segmentation_dataset, tmp_path):
    import json

    output = tmp_path / "coco"

    report = convert_dataset(yolo_segmentation_dataset, output, "yolo", "coco", "segmentation")

    assert report.success
    data = json.loads((output / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert data["annotations"][0]["bbox"] == [1.0, 1.0, 7.0, 7.0]
    assert data["annotations"][0]["segmentation"] == [[1.0, 1.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0]]


def test_invalid_yolo_label_does_not_write_output(yolo_detection_dataset, tmp_path):
    label_path = yolo_detection_dataset / "labels" / "train" / "image1.txt"
    label_path.write_text("0 0.5 0.5\n", encoding="utf-8")
    output = tmp_path / "coco"

    report = convert_dataset(yolo_detection_dataset, output, "yolo", "coco", "detection")

    assert not report.success
    assert any(item.code == "YOLO_READ_FAILED" for item in report.validation.errors)
    assert not output.exists()
