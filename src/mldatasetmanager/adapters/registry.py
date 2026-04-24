from mldatasetmanager.adapters.base import DatasetAdapter
from mldatasetmanager.adapters.coco import CocoAdapter
from mldatasetmanager.adapters.yolo import YoloAdapter

_ADAPTERS: dict[str, DatasetAdapter] = {
    "coco": CocoAdapter(),
    "yolo": YoloAdapter(),
}


def get_adapter(format_name: str) -> DatasetAdapter:
    normalized = format_name.lower().strip()
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unsupported format '{format_name}'. Supported formats: {supported}"
        ) from exc
