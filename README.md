# MLDatasetManager

Dataset manager for machine learning annotation workflows.

The initial setup supports a production-oriented first path:

- detection and instance-segmentation datasets
- validate images, categories, references, boxes, and polygons
- convert COCO detection datasets to YOLO detection
- convert COCO polygon segmentation datasets to YOLO segmentation
- convert YOLO detection datasets to COCO detection
- convert YOLO polygon segmentation datasets to COCO segmentation
- generate JSON reports for validation and conversion

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ruff check
ruff format --check
pytest
```

## CLI

```powershell
mldm inspect CacaoMaturityDetection.coco\train --format coco
mldm validate CacaoMaturityDetection.coco\train --format coco --report validation.json
mldm convert CacaoMaturityDetection.coco\train out-yolo --from coco --to yolo --task detection
mldm convert out-yolo out-coco --from yolo --to coco --task detection
```

Conversion fails by default if validation finds errors. Existing output directories are not
overwritten unless `--overwrite` is passed.

## Python API

```python
from mldatasetmanager import convert_dataset, import_dataset, validate_dataset

dataset = import_dataset("CacaoMaturityDetection.coco/train", format="coco")
report = validate_dataset(dataset)
conversion = convert_dataset(
    "CacaoMaturityDetection.coco/train",
    "out-yolo",
    source_format="coco",
    target_format="yolo",
    task="detection",
)
```

## Docker

Docker is provided for reproducible checks:

```powershell
docker build -t mldatasetmanager .
docker run --rm mldatasetmanager
```
