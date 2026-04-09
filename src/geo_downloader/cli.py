from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from geo_downloader.workflow import GeoDownloadWorkflow, WorkflowSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download-geo",
        description="Workflow-oriented downloader for GEO single-cell series.",
    )
    parser.add_argument("accession", help="GEO accession, for example GSE150290")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads"),
        help="Directory where accession folders will be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print download targets without downloading files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress workflow step prefixes.",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Suppress the detailed per-accession summary section.",
    )
    parser.add_argument(
        "--write-graph",
        type=Path,
        help="Write a simple Mermaid workflow graph to a file.",
    )
    return parser


def _write_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "flowchart TD",
                "  A[Resolve GEO page] --> B[Summarize artifacts]",
                "  B --> C[Download files]",
                "  C --> D[Write manifest and report]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.write_graph:
        _write_graph(args.write_graph)

    workflow = GeoDownloadWorkflow(
        WorkflowSettings(no_progress=args.no_progress, no_summaries=args.no_summaries)
    )
    try:
        workflow.run(args.accession, args.output_dir, dry_run=args.dry_run)
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

