from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mldatasetmanager.api import convert_dataset, import_dataset, validate_dataset
from mldatasetmanager.reports import ConversionReport, ValidationReport

app = typer.Typer(no_args_is_help=True)


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    format: Annotated[str, typer.Option("--format")] = "coco",
) -> None:
    dataset = import_dataset(path, format=format)
    typer.echo(f"Dataset: {dataset.name}")
    typer.echo(f"Format: {format}")
    typer.echo(f"Task: {dataset.dataset_type.value}")
    typer.echo(f"Images: {len(dataset.images)}")
    typer.echo(f"Annotations: {len(dataset.annotations)}")
    typer.echo(f"Categories: {len(dataset.classes)}")


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    format: Annotated[str, typer.Option("--format")] = "coco",
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    validation_report = validate_dataset(path, format=format)
    _write_validation_report(validation_report, report)
    if validation_report.has_errors:
        raise typer.Exit(code=1)


@app.command()
def convert(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    output: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True)],
    source_format: Annotated[str, typer.Option("--from")] = "coco",
    target_format: Annotated[str, typer.Option("--to")] = "yolo",
    task: Annotated[str, typer.Option("--task")] = "detection",
    report: Annotated[Path | None, typer.Option("--report")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    conversion_report = convert_dataset(
        source,
        output,
        source_format=source_format,
        target_format=target_format,
        task=task,
        options={"overwrite": overwrite},
    )
    _write_conversion_report(conversion_report, report)
    if not conversion_report.success:
        raise typer.Exit(code=1)


def _write_validation_report(validation_report: ValidationReport, report_path: Path | None) -> None:
    typer.echo(
        f"Validation: {len(validation_report.errors)} errors, "
        f"{len(validation_report.warnings)} warnings"
    )
    if report_path is not None:
        validation_report.write_json(report_path)
        typer.echo(f"Report: {report_path}")


def _write_conversion_report(conversion_report: ConversionReport, report_path: Path | None) -> None:
    typer.echo(
        f"Conversion: {'success' if conversion_report.success else 'failed'}; "
        f"{len(conversion_report.validation.errors)} errors, "
        f"{len(conversion_report.validation.warnings)} warnings; "
        f"{conversion_report.files_written} files written"
    )
    if report_path is not None:
        conversion_report.write_json(report_path)
        typer.echo(f"Report: {report_path}")


if __name__ == "__main__":
    app()
