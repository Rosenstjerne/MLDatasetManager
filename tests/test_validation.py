import json

from mldatasetmanager.adapters.coco import CocoAdapter
from mldatasetmanager.validation.validators import validate_dataset


def test_validation_reports_missing_image(coco_detection_dataset):
    (coco_detection_dataset / "image1.jpg").unlink()
    dataset = CocoAdapter().read(coco_detection_dataset)

    report = validate_dataset(dataset)

    assert report.has_errors
    assert any(item.code == "MISSING_IMAGE" for item in report.errors)


def test_validation_reports_unknown_category(coco_detection_dataset):
    annotation_path = coco_detection_dataset / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"][0]["category_id"] = 99
    annotation_path.write_text(json.dumps(data), encoding="utf-8")
    dataset = CocoAdapter().read(coco_detection_dataset)

    report = validate_dataset(dataset)

    assert report.has_errors
    assert any(item.code == "UNKNOWN_CATEGORY_REFERENCE" for item in report.errors)


def test_validation_reports_unknown_image_reference(coco_detection_dataset):
    annotation_path = coco_detection_dataset / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"][0]["image_id"] = 99
    annotation_path.write_text(json.dumps(data), encoding="utf-8")
    dataset = CocoAdapter().read(coco_detection_dataset)

    report = validate_dataset(dataset)

    assert report.has_errors
    assert any(item.code == "UNKNOWN_IMAGE_REFERENCE" for item in report.errors)


def test_validation_reports_out_of_bounds_bbox(coco_detection_dataset):
    annotation_path = coco_detection_dataset / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"][0]["bbox"] = [1, 1, 20, 20]
    annotation_path.write_text(json.dumps(data), encoding="utf-8")
    dataset = CocoAdapter().read(coco_detection_dataset)

    report = validate_dataset(dataset)

    assert report.has_errors
    assert any(item.code == "BBOX_OUT_OF_BOUNDS" for item in report.errors)


def test_path_validation_reports_invalid_polygon(coco_detection_dataset):
    from mldatasetmanager.api import validate_dataset as validate_path

    annotation_path = coco_detection_dataset / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data["annotations"][0]["segmentation"] = [[1, 1, 2, 2]]
    annotation_path.write_text(json.dumps(data), encoding="utf-8")

    report = validate_path(coco_detection_dataset, format="coco")

    assert report.has_errors
    assert any(item.code == "DATASET_READ_FAILED" for item in report.errors)


def test_structure_validation_reports_corrupt_json(coco_detection_dataset):
    report = CocoAdapter().validate_structure(coco_detection_dataset)
    assert not report.has_errors

    (coco_detection_dataset / "_annotations.coco.json").write_text("{", encoding="utf-8")
    report = CocoAdapter().validate_structure(coco_detection_dataset)

    assert report.has_errors
    assert any(item.code == "COCO_JSON_INVALID" for item in report.errors)
