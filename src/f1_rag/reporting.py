from __future__ import annotations

import json
import unicodedata
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle


def _pdf_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).replace("−", "-"))
    return normalized.encode("latin-1", "replace").decode("latin-1")


def dataframe_to_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_pdf(frame: pd.DataFrame, title: str) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A3),
        leftMargin=24,
        rightMargin=24,
        topMargin=28,
        bottomMargin=24,
        title=title,
    )
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "ReportTableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "ReportTableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    available_width = document.width
    columns = len(frame.columns)
    if not columns:
        raise ValueError("A PDF report requires at least one table column.")
    column_width = available_width / columns
    table_data = [[Paragraph(escape(_pdf_text(value)), header_style) for value in frame.columns]]
    for row in frame.itertuples(index=False, name=None):
        table_data.append([
            Paragraph(escape("" if pd.isna(value) else _pdf_text(value)), cell_style)
            for value in row
        ])
    table = LongTable(
        table_data,
        colWidths=[column_width] * columns,
        repeatRows=1,
        splitByRow=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f4f8")]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd2dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    document.build([
        Paragraph(escape(_pdf_text(title)), styles["Title"]),
        Spacer(1, 12),
        table,
    ])
    return output.getvalue()


def save_evaluation_run(
    output_dir: Path,
    *,
    test_type: str,
    dataset: str,
    configuration: dict[str, Any],
    rows: list[dict[str, Any]],
    errors: list[str] | None = None,
) -> dict[str, Any]:
    if not test_type.strip() or not dataset.strip():
        raise ValueError("test_type and dataset must not be empty.")
    record = {
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "test_type": test_type,
        "dataset": dataset,
        "configuration": configuration,
        "rows": rows,
        "errors": errors or [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluations.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return record


def load_evaluation_runs(output_dir: Path, dataset: str | None = None) -> list[dict[str, Any]]:
    path = output_dir / "evaluations.jsonl"
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid evaluation history at line {line_number}: {error}") from error
            if (
                not isinstance(record, dict)
                or not all(
                    isinstance(record.get(key), str)
                    for key in ("run_id", "created_at", "test_type", "dataset")
                )
                or not isinstance(record.get("configuration"), dict)
                or not isinstance(record.get("rows"), list)
                or any(not isinstance(row, dict) for row in record["rows"])
                or not isinstance(record.get("errors"), list)
                or any(not isinstance(error, str) for error in record["errors"])
            ):
                raise ValueError(f"Invalid evaluation history record at line {line_number}.")
            if dataset is None or record.get("dataset") == dataset:
                records.append(record)
    return list(reversed(records))
