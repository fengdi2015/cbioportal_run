from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from geo_downloader.h5ad import (
    convert_all_sample_bundles_to_h5ad,
    convert_sample_bundle_to_h5ad,
    list_sample_bundles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo-to-h5ad",
        description="Convert a GEO raw tar bundle into an AnnData .h5ad file.",
    )
    parser.add_argument("archive", type=Path, help="Path to the downloaded GEO raw tar file.")
    parser.add_argument(
        "--sample",
        help="Sample bundle name, for example GSM4546300_Pat01-A. Defaults to the first bundle in the archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .h5ad file path. Defaults to downloads/<sample>.h5ad next to the archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for --all-samples output. Defaults to the archive directory.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List sample bundles and exit.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Write one .h5ad file per detected sample bundle.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        bundles = list_sample_bundles(args.archive)
        if args.list_samples:
            for bundle in bundles:
                print(bundle.sample_id)
            return 0
        if not bundles:
            raise ValueError("no sample bundles found in archive")
        if args.all_samples:
            output_dir = args.output_dir or args.output or args.archive.parent
            written = convert_all_sample_bundles_to_h5ad(args.archive, output_dir)
            for path in written:
                print(f"wrote {path}")
            return 0
        sample_id = args.sample or bundles[0].sample_id
        if args.sample is None and len(bundles) > 1:
            available = ", ".join(bundle.sample_id for bundle in bundles[:10])
            raise ValueError(
                f"archive contains multiple sample bundles. use --sample or --all-samples. available examples: {available}"
            )
        output = args.output or args.archive.with_name(f"{sample_id}.h5ad")
        convert_sample_bundle_to_h5ad(args.archive, sample_id, output)
        print(f"wrote {output}")
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
