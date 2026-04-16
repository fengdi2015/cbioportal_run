from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from geo_downloader.cbioportal import (
    CbioPortalClient,
    build_cbioportal_study_index,
    default_cbioportal_study_index_path,
    save_cbioportal_study_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbioportal-build-index",
        description="Build a frozen cBioPortal cancer study index for deterministic cancer-family queries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_cbioportal_study_index_path(),
        help="Where to write the JSON index. Defaults to the bundled package index file.",
    )
    parser.add_argument(
        "--base-url",
        default="https://www.cbioportal.org",
        help="cBioPortal base URL, useful for private instances.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    client = CbioPortalClient(base_url=args.base_url)
    index = build_cbioportal_study_index(client)
    output_path = save_cbioportal_study_index(index, args.output)
    print(f"wrote: {output_path}")
    print(f"entries: {len(index.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
