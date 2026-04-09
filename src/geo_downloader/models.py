from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GeoArtifact:
    filename: str
    url: str


@dataclass(frozen=True)
class GeoSeriesRecord:
    accession: str
    page_url: str
    related_sra_accessions: tuple[str, ...] = ()
    artifacts: tuple[GeoArtifact, ...] = ()


@dataclass(frozen=True)
class DownloadedArtifact:
    artifact: GeoArtifact
    path: Path
    bytes_downloaded: int = 0


@dataclass(frozen=True)
class WorkflowResult:
    record: GeoSeriesRecord
    output_dir: Path
    files_dir: Path
    manifest_path: Path
    report_path: Path
    downloaded_artifacts: tuple[DownloadedArtifact, ...] = field(default_factory=tuple)

