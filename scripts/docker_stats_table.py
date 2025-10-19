"""Utility to transform `docker stats` output into readable tables.

The script runs ``docker stats --no-stream`` with a JSON format so the
statistics can be parsed reliably.  It then converts the measurements into
percentages and optionally exports them as plain text, Markdown, CSV, or JSON.

Examples
--------

To display a simple plain-text table in the terminal::

    python scripts/docker_stats_table.py

To export the table to Markdown and save it to a file::

    python scripts/docker_stats_table.py --format markdown --output stats.md

If Docker is not available or no containers are running, the script exits
gracefully with a descriptive error message.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class TableRow:
    """Structure holding the processed statistics for a container."""

    container_id: str
    name: str
    cpu_percent: float
    mem_percent: float
    mem_usage: str
    net_io: str
    block_io: str
    pids: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "Container": self.container_id,
            "Name": self.name,
            "CPU %": f"{self.cpu_percent:.2f}",
            "Mem %": f"{self.mem_percent:.2f}",
            "Mem Usage": self.mem_usage,
            "Net I/O": self.net_io,
            "Block I/O": self.block_io,
            "PIDs": self.pids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a table of CPU, memory and I/O statistics for running Docker "
            "containers."
        )
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("plain", "markdown", "csv", "json"),
        default="plain",
        help="Output format for the table (default: plain).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to save the generated table. When omitted the table is printed.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="When using CSV format, omit the header row.",
    )
    return parser.parse_args()


def fetch_docker_stats() -> List[Dict[str, str]]:
    """Return the parsed JSON objects from ``docker stats``.

    The command ``docker stats --no-stream --format '{{json .}}'`` outputs one
    JSON object per container, each describing the current resource utilisation.
    """

    try:
        completed = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - triggered when docker missing
        raise SystemExit("Docker is not installed or not available in PATH.") from exc

    if completed.returncode != 0:
        message = completed.stderr.strip() or "docker stats failed."
        raise SystemExit(message)

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No running containers found.")

    stats = []
    for line in lines:
        try:
            stats.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Unexpected docker stats format: {line}") from exc
    return stats


def parse_percent(value: str) -> float:
    """Convert a string percentage like ``'12.4%'`` into a float."""

    try:
        return float(value.strip().rstrip("%"))
    except (AttributeError, ValueError):
        return 0.0


def build_rows(raw_stats: Iterable[Dict[str, str]]) -> List[TableRow]:
    rows: List[TableRow] = []
    for entry in raw_stats:
        rows.append(
            TableRow(
                container_id=entry.get("Container", ""),
                name=entry.get("Name", ""),
                cpu_percent=parse_percent(entry.get("CPUPerc", "0")),
                mem_percent=parse_percent(entry.get("MemPerc", "0")),
                mem_usage=entry.get("MemUsage", ""),
                net_io=entry.get("NetIO", ""),
                block_io=entry.get("BlockIO", ""),
                pids=entry.get("PIDs", ""),
            )
        )
    return rows


def format_plain_table(rows: Sequence[TableRow]) -> str:
    dictionaries = [row.as_dict() for row in rows]
    headers = list(dictionaries[0].keys())
    widths = {header: len(header) for header in headers}
    for row in dictionaries:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    def render_line(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[header]) for value, header in zip(values, headers))

    divider = "-+-".join("-" * widths[header] for header in headers)

    lines = [render_line(headers), divider]
    for row in dictionaries:
        lines.append(render_line([row[header] for header in headers]))
    return "\n".join(lines)


def format_markdown_table(rows: Sequence[TableRow]) -> str:
    dictionaries = [row.as_dict() for row in rows]
    headers = list(dictionaries[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = [
        "| " + " | ".join(row[header] for header in headers) + " |"
        for row in dictionaries
    ]
    return "\n".join([header_line, separator_line, *data_lines])


def format_csv(rows: Sequence[TableRow], include_header: bool = True) -> str:
    import csv

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].as_dict().keys()))
    if include_header:
        writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())
    return output.getvalue().rstrip()


def format_json(rows: Sequence[TableRow]) -> str:
    return json.dumps([row.as_dict() for row in rows], indent=2)


def render_table(rows: Sequence[TableRow], fmt: str, include_header: bool) -> str:
    if not rows:
        return "No rows to display."

    if fmt == "plain":
        return format_plain_table(rows)
    if fmt == "markdown":
        return format_markdown_table(rows)
    if fmt == "csv":
        return format_csv(rows, include_header=include_header)
    if fmt == "json":
        return format_json(rows)
    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    args = parse_args()
    raw_stats = fetch_docker_stats()
    rows = build_rows(raw_stats)
    output = render_table(rows, args.format, include_header=not args.no_header)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
