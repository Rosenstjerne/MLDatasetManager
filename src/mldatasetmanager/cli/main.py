from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mldatasetmanager.api import (
    convert_dataset,
    import_dataset,
    merge_datasets,
    resplit_dataset,
    validate_dataset,
)
from mldatasetmanager.reports import ConversionReport, MergeReport, SplitReport, ValidationReport

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
    allow_lossy: Annotated[bool, typer.Option("--allow-lossy")] = False,
) -> None:
    conversion_report = convert_dataset(
        source,
        output,
        source_format=source_format,
        target_format=target_format,
        task=task,
        options={"overwrite": overwrite, "allow_lossy": allow_lossy},
    )
    _write_conversion_report(conversion_report, report)
    if not conversion_report.success:
        raise typer.Exit(code=1)


@app.command()
def merge(
    output: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True)],
    sources: Annotated[list[Path], typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    formats: Annotated[str, typer.Option("--formats")] = "coco",
    target_format: Annotated[str, typer.Option("--to")] = "coco",
    task: Annotated[str, typer.Option("--task")] = "detection",
    class_policy: Annotated[str, typer.Option("--class-policy")] = "union",
    report: Annotated[Path | None, typer.Option("--report")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    source_formats = [item.strip() for item in formats.split(",") if item.strip()]
    merge_report = merge_datasets(
        sources,
        output,
        source_formats=source_formats,
        target_format=target_format,
        task=task,
        options={"overwrite": overwrite, "class_policy": class_policy},
    )
    _write_merge_report(merge_report, report)
    if not merge_report.success:
        raise typer.Exit(code=1)


@app.command("split")
def split_dataset_command(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    output: Annotated[Path, typer.Argument(file_okay=False, dir_okay=True)],
    format: Annotated[str, typer.Option("--format")] = "coco",
    target_format: Annotated[str | None, typer.Option("--to")] = None,
    task: Annotated[str, typer.Option("--task")] = "detection",
    ratios: Annotated[str, typer.Option("--ratios")] = "80,10,10",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    stratify: Annotated[str, typer.Option("--stratify")] = "none",
    report: Annotated[Path | None, typer.Option("--report")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    try:
        parsed_ratios = [float(item.strip()) for item in ratios.split(",") if item.strip()]
    except ValueError as exc:
        raise typer.BadParameter("--ratios must be comma-separated numbers") from exc
    split_report = resplit_dataset(
        source,
        output,
        source_format=format,
        target_format=target_format or format,
        task=task,
        ratios=parsed_ratios,
        seed=seed,
        stratify=stratify,
        options={"overwrite": overwrite},
    )
    _write_split_report(split_report, report)
    if not split_report.success:
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


def _write_merge_report(merge_report: MergeReport, report_path: Path | None) -> None:
    typer.echo(
        f"Merge: {'success' if merge_report.success else 'failed'}; "
        f"{len(merge_report.validation.errors)} errors, "
        f"{len(merge_report.validation.warnings)} warnings; "
        f"{merge_report.files_written} files written"
    )
    if report_path is not None:
        merge_report.write_json(report_path)
        typer.echo(f"Report: {report_path}")


def _write_split_report(split_report: SplitReport, report_path: Path | None) -> None:
    typer.echo(
        f"Split: {'success' if split_report.success else 'failed'}; "
        f"{len(split_report.validation.errors)} errors, "
        f"{len(split_report.validation.warnings)} warnings; "
        f"{split_report.files_written} files written"
    )
    if split_report.summary.get("splits"):
        typer.echo(f"Splits: {split_report.summary['splits']}")
    if report_path is not None:
        split_report.write_json(report_path)
        typer.echo(f"Report: {report_path}")


if __name__ == "__main__":
    app()
