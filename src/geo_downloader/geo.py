from __future__ import annotations

import html.parser
import re
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from geo_downloader.models import GeoArtifact, GeoSeriesRecord


GEO_SERIES_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
GEO_DOWNLOAD_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&format=file"


class _LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value for key, value in attrs if value}
        href = attr_map.get("href")
        if href:
            self.links.append((href, attr_map.get("title") or ""))


def geo_series_page_url(accession: str) -> str:
    return GEO_SERIES_URL.format(accession=accession)


def geo_series_download_url(accession: str) -> str:
    return GEO_DOWNLOAD_URL.format(accession=accession)


def fetch_text(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def build_geo_record(accession: str, html_text: str) -> GeoSeriesRecord:
    accession = accession.strip().upper()
    parser = _LinkParser()
    parser.feed(html_text)

    artifacts: list[GeoArtifact] = []
    seen: set[str] = set()
    for href, _title in parser.links:
        absolute = urljoin(geo_series_page_url(accession), href)
        if "geo/download/" not in absolute and "ftp.ncbi.nlm.nih.gov/geo/" not in absolute:
            continue
        if _looks_like_directory(absolute):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        artifacts.append(GeoArtifact(filename=_filename_from_download_url(absolute, accession), url=absolute))

    if not artifacts:
        artifacts.append(
            GeoArtifact(
                filename=f"{accession}_RAW.tar",
                url=geo_series_download_url(accession),
            )
        )

    related_sra = tuple(sorted(set(re.findall(r"\bSRP\d+\b", html_text))))
    return GeoSeriesRecord(
        accession=accession,
        page_url=geo_series_page_url(accession),
        related_sra_accessions=related_sra,
        artifacts=tuple(artifacts),
    )


def _filename_from_download_url(url: str, accession: str) -> str:
    parsed = urlparse(url)
    if "geo/download" in parsed.path:
        query = parse_qs(parsed.query)
        if query.get("acc", [None])[0]:
            return f"{accession}_RAW.tar"

    match = re.search(r"/([^/?#]+)(?:\?|$)", parsed.path)
    if match:
        return match.group(1)
    return "download.bin"


def _looks_like_directory(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        return True
    name = path.rsplit("/", 1)[-1]
    return "." not in name and parsed.query == ""
