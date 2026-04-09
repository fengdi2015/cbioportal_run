from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from geo_downloader.geo import build_geo_record, fetch_text
from geo_downloader.models import DownloadedArtifact, GeoSeriesRecord, WorkflowResult
from geo_downloader.report import write_manifest, write_report


@dataclass(frozen=True)
class WorkflowSettings:
    no_progress: bool = False
    no_summaries: bool = False


def download_file(url: str, destination: Path, timeout: int = 60) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    total = 0
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
    return total


class GeoDownloadWorkflow:
    def __init__(self, settings: WorkflowSettings | None = None) -> None:
        self.settings = settings or WorkflowSettings()

    def run(self, accession: str, output_dir: Path, dry_run: bool = False) -> WorkflowResult:
        accession = accession.strip().upper()
        accession_dir = output_dir / accession
        files_dir = accession_dir / "files"

        self._say_step(1, 4, "Resolve GEO series page")
        page_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        html_text = fetch_text(page_url)
        record = build_geo_record(accession, html_text)

        self._say_step(2, 4, "Summarize downloadable artifacts")
        if not self.settings.no_summaries:
            self._print_summary(record)

        downloaded: list[DownloadedArtifact] = []
        self._say_step(3, 4, "Download files")
        for artifact in record.artifacts:
            destination = files_dir / artifact.filename
            if dry_run:
                self._print_download_target(artifact.filename, artifact.url, destination)
                continue
            bytes_downloaded = download_file(artifact.url, destination)
            downloaded.append(
                DownloadedArtifact(
                    artifact=artifact,
                    path=destination,
                    bytes_downloaded=bytes_downloaded,
                )
            )
            self._print_download_target(artifact.filename, artifact.url, destination, bytes_downloaded)

        self._say_step(4, 4, "Write manifest and report")
        manifest_path = write_manifest(record, accession_dir, downloaded)
        report_path = write_report(record, accession_dir, downloaded)
        return WorkflowResult(
            record=record,
            output_dir=accession_dir,
            files_dir=files_dir,
            manifest_path=manifest_path,
            report_path=report_path,
            downloaded_artifacts=tuple(downloaded),
        )

    def _say_step(self, step: int, total: int, title: str) -> None:
        if self.settings.no_progress:
            print(title)
            return
        print(f"[{step}/{total}] {title}")

    def _print_summary(self, record: GeoSeriesRecord) -> None:
        print(f"GEO series: {record.accession}")
        print(f"Page: {record.page_url}")
        if record.related_sra_accessions:
            print(f"Related SRA accession(s): {', '.join(record.related_sra_accessions)}")
        for artifact in record.artifacts:
            print(f"Artifact: {artifact.filename}")
            print(f"  URL: {artifact.url}")

    def _print_download_target(
        self,
        filename: str,
        url: str,
        destination: Path,
        bytes_downloaded: int | None = None,
    ) -> None:
        print(f"  {filename}")
        print(f"    URL: {url}")
        print(f"    Destination: {destination}")
        if bytes_downloaded is not None:
            print(f"    Bytes: {bytes_downloaded}")

