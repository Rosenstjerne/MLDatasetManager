from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetTask(StrEnum):
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance-segmentation"
    ORIENTED_DETECTION = "oriented-detection"


class AxisAlignedBBox(BaseModel):
    kind: Literal["bbox"] = "bbox"
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def validate_extent(self) -> AxisAlignedBBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bbox must have positive width and height")
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class Polygon(BaseModel):
    kind: Literal["polygon"] = "polygon"
    points: list[tuple[float, float]]

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) < 3:
            raise ValueError("polygon requires at least 3 points")
        return points


class OrientedBBox(BaseModel):
    kind: Literal["obb"] = "obb"
    points: list[tuple[float, float]]

    @field_validator("points")
    @classmethod
    def validate_points(cls, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) != 4:
            raise ValueError("oriented bbox requires exactly 4 points")
        return points


class MultiPolygon(BaseModel):
    kind: Literal["multipolygon"] = "multipolygon"
    polygons: list[Polygon]

    @field_validator("polygons")
    @classmethod
    def validate_polygons(cls, polygons: list[Polygon]) -> list[Polygon]:
        if not polygons:
            raise ValueError("multipolygon requires at least one polygon")
        return polygons


class RLEMask(BaseModel):
    kind: Literal["rle"] = "rle"
    size: tuple[int, int]
    counts: str | list[int]


Geometry = Annotated[
    AxisAlignedBBox | OrientedBBox | Polygon | MultiPolygon | RLEMask,
    Field(discriminator="kind"),
]


class Category(BaseModel):
    id: int
    name: str
    supercategory: str | None = None
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageAsset(BaseModel):
    id: int | str
    path: Path
    width: int
    height: int
    checksum: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ImageAsset:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        return self


class Annotation(BaseModel):
    id: int | str
    image_id: int | str
    category_id: int
    task_type: DatasetTask
    geometry: Geometry
    attributes: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    dataset_type: DatasetTask
    root: Path
    classes: list[Category]
    images: list[ImageAsset]
    annotations: list[Annotation]
    splits: dict[str, list[int | str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def image_by_id(self) -> dict[int | str, ImageAsset]:
        return {image.id: image for image in self.images}

    @property
    def category_by_id(self) -> dict[int, Category]:
        return {category.id: category for category in self.classes}
