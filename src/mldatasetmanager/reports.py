from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]


class Diagnostic(BaseModel):
    severity: Severity
    code: str
    message: str
    file_path: str | None = None
    image_id: int | str | None = None
    annotation_id: int | str | None = None
    category_id: int | str | None = None


class ValidationReport(BaseModel):
    dataset_path: str
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [item for item in self.diagnostics if item.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        **context: int | str | None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(severity=severity, code=code, message=message, **context)
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


class ConversionReport(BaseModel):
    source_path: str
    output_path: str
    source_format: str
    target_format: str
    task: str
    validation: ValidationReport
    files_written: int = 0
    skipped_files: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.validation.has_errors

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
