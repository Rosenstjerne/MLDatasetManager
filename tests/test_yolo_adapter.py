from mldatasetmanager.adapters.yolo import YoloAdapter
from mldatasetmanager.core.models import AxisAlignedBBox, DatasetTask, MultiPolygon


def test_yolo_reader_loads_detection_dataset(yolo_detection_dataset):
    dataset = YoloAdapter().read(yolo_detection_dataset)

    assert dataset.dataset_type == DatasetTask.DETECTION
    assert len(dataset.images) == 1
    assert len(dataset.annotations) == 1
    assert isinstance(dataset.annotations[0].geometry, AxisAlignedBBox)


def test_yolo_reader_loads_segmentation_dataset(yolo_segmentation_dataset):
    dataset = YoloAdapter().read(yolo_segmentation_dataset, {"task": "segmentation"})

    assert dataset.dataset_type == DatasetTask.INSTANCE_SEGMENTATION
    assert len(dataset.images) == 1
    assert len(dataset.annotations) == 1
    assert isinstance(dataset.annotations[0].geometry, MultiPolygon)
