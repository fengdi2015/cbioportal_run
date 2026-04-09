from __future__ import annotations

import json
from pathlib import Path

from geo_downloader.models import DownloadedArtifact, GeoSeriesRecord


def write_manifest(record: GeoSeriesRecord, output_dir: Path, downloaded: list[DownloadedArtifact]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    payload = {
        "accession": record.accession,
        "page_url": record.page_url,
        "related_sra_accessions": list(record.related_sra_accessions),
        "artifacts": [{"filename": artifact.filename, "url": artifact.url} for artifact in record.artifacts],
        "downloaded_artifacts": [
            {
                "filename": item.artifact.filename,
                "url": item.artifact.url,
                "path": str(item.path),
                "bytes_downloaded": item.bytes_downloaded,
            }
            for item in downloaded
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_report(record: GeoSeriesRecord, output_dir: Path, downloaded: list[DownloadedArtifact]) -> Path:
    report_path = output_dir / "report.txt"
    lines = [
        f"Accession: {record.accession}",
        f"Page: {record.page_url}",
    ]
    if record.related_sra_accessions:
        lines.append(f"Related SRA accession(s): {', '.join(record.related_sra_accessions)}")
    lines.append("")
    lines.append("Artifacts:")
    for artifact in record.artifacts:
        lines.append(f"- {artifact.filename}")
        lines.append(f"  {artifact.url}")
    if downloaded:
        lines.append("")
        lines.append("Downloaded:")
        for item in downloaded:
            lines.append(f"- {item.path.name} ({item.bytes_downloaded} bytes)")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
