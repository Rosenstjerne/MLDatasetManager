from mldatasetmanager.adapters.coco import CocoAdapter
from mldatasetmanager.core.models import DatasetTask, MultiPolygon


def test_coco_reader_loads_detection_dataset(coco_detection_dataset):
    dataset = CocoAdapter().read(coco_detection_dataset)

    assert dataset.dataset_type == DatasetTask.DETECTION
    assert len(dataset.images) == 1
    assert len(dataset.annotations) == 1
    assert dataset.classes[0].name == "object"


def test_coco_reader_loads_polygon_segmentation_dataset(coco_segmentation_dataset):
    dataset = CocoAdapter().read(coco_segmentation_dataset)

    assert dataset.dataset_type == DatasetTask.INSTANCE_SEGMENTATION
    assert isinstance(dataset.annotations[0].geometry, MultiPolygon)
