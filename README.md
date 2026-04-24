# MLDatasetManager

Dataset manager for machine learning annotation workflows.

The initial setup supports a production-oriented first path:

- detection and instance-segmentation datasets
- oriented bounding box datasets in YOLO OBB format
- validate images, categories, references, boxes, and polygons
- convert COCO detection datasets to YOLO detection
- convert COCO polygon segmentation datasets to YOLO segmentation
- convert YOLO detection datasets to COCO detection
- convert YOLO polygon segmentation datasets to COCO segmentation
- convert YOLO OBB datasets to COCO polygon segmentation
- convert COCO detection boxes or quadrilateral polygons to YOLO OBB
- merge multiple COCO/YOLO datasets into a COCO or YOLO output dataset
- resplit datasets into train/val/test outputs with deterministic seeds
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
mldm convert out-yolo-obb out-coco-obb --from yolo --to coco --task obb
mldm convert out-yolo-obb out-yolo-det --from yolo --to yolo --task detection --allow-lossy
mldm merge merged-coco CacaoMaturityDetection.coco\train out-yolo --formats coco,yolo --to coco --task detection
mldm split CacaoMaturityDetection.coco\train split-yolo --format coco --to yolo --task detection --ratios 80,10,10 --seed 42
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

Merging supports `--class-policy union` by default. Use `--class-policy exact` when
secondary datasets must not introduce new class names.

Splitting writes split-aware outputs. YOLO outputs use `images/train`, `images/val`,
`images/test` and matching label folders. COCO outputs use `train`, `val`, and `test`
folders, each with its own `_annotations.coco.json`. Use `--stratify class` to
interleave image-level primary classes before assigning split counts.

YOLO OBB labels use normalized four-corner rows:

```text
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

COCO has no native OBB schema in this tool; OBB export to COCO is represented as
polygon segmentation. OBB-to-axis-aligned detection output is lossy and requires
`--allow-lossy`.

## Docker

Docker is provided for reproducible checks:

```powershell
docker build -t mldatasetmanager .
docker run --rm mldatasetmanager
```
