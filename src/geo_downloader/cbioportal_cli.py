from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from geo_downloader.cbioportal import (
    CbioPortalClient,
    build_cbioportal_data_source,
    compute_mutation_expression_associations,
    default_cbioportal_study_index_path,
    load_cbioportal_study_index,
    plot_mutation_expression_forest_summary,
    plot_mutation_expression_violin_panels,
    plot_cbioportal_oncoplot,
    save_cbioportal_outputs,
    resolve_all_studies_from_index,
    _is_mutated,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbioportal-plot",
        description="Download cBioPortal mutation and mRNA data and draw an oncoplot-style heatmap.",
    )
    parser.add_argument("--study-id", help="cBioPortal study ID, for example brca_tcga_pan_can_atlas_2018")
    parser.add_argument(
        "--cancer",
        help="Cancer family name or code. Runs the default TCGA GDC 2025 study and, if present, a CPTAC study for the same cancer.",
    )
    parser.add_argument(
        "--all-cancers",
        action="store_true",
        help="Run every cancer family from the frozen index, using TCGA GDC 2025 and matching CPTAC studies when available. Studies with no usable mutation signal are skipped.",
    )
    parser.add_argument(
        "--mutation-genes",
        help="Comma-separated HUGO gene symbols to plot as mutation rows.",
    )
    parser.add_argument(
        "--mrna-genes",
        help="Comma-separated HUGO gene symbols to plot as mRNA rows. Defaults to --mutation-genes.",
    )
    parser.add_argument(
        "--sample-list-id",
        help="Sample list ID to use. Defaults to the study's all-cases list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "cbioportal",
        help="Directory where tables and plots will be written.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image file path for the violin figure. Defaults to <output-dir>/<study-id>/violin.png",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Use a compact presentation layout with thin sample columns and sparse sample labels.",
    )
    parser.add_argument(
        "--figure-style",
        choices=("heatmap", "relationship"),
        default="relationship",
        help="Figure style to render. Relationship mode uses violin+jitter panels plus a forest summary and also writes a heatmap.",
    )
    parser.add_argument(
        "--max-sample-labels",
        type=int,
        default=20,
        help="Maximum number of sample labels to show in presentation mode.",
    )
    parser.add_argument(
        "--base-url",
        default="https://www.cbioportal.org",
        help="cBioPortal base URL, useful for private instances.",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=default_cbioportal_study_index_path(),
        help="Frozen cBioPortal study index JSON used to resolve --cancer deterministically.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    mutation_genes = [item.strip() for item in args.mutation_genes.split(",") if item.strip()] if args.mutation_genes else []
    mrna_genes = [item.strip() for item in args.mrna_genes.split(",") if item.strip()] if args.mrna_genes else None
    if args.all_cancers and (args.study_id or args.cancer):
        print("error: specify --all-cancers by itself", file=sys.stderr)
        return 1
    if args.study_id and args.cancer:
        print("error: specify either --study-id or --cancer, not both", file=sys.stderr)
        return 1
    if not args.study_id and not args.cancer and not args.all_cancers:
        print("error: specify --study-id, --cancer, or --all-cancers", file=sys.stderr)
        return 1
    if not mutation_genes and not mrna_genes:
        print("error: at least one mutation or mRNA gene is required", file=sys.stderr)
        return 1

    client = CbioPortalClient(base_url=args.base_url)
    try:
        study_selection = None
        study_selections = None
        if args.cancer:
            if not args.index_file.exists():
                print(f"error: index file not found at {args.index_file}", file=sys.stderr)
                return 1
            study_index = load_cbioportal_study_index(args.index_file)
            study_selection = client.resolve_default_studies_for_cancer(args.cancer, study_index=study_index)
        elif args.all_cancers:
            if not args.index_file.exists():
                print(f"error: index file not found at {args.index_file}", file=sys.stderr)
                return 1
            study_index = load_cbioportal_study_index(args.index_file)
            study_selections = resolve_all_studies_from_index(study_index)
        study_jobs: list[tuple[str, Any | None]] = []
        if args.study_id:
            study_jobs = [(args.study_id, None)]
        elif args.cancer:
            study_ids = list(study_selection.study_ids if study_selection is not None else ())
            study_records = list(study_selection.study_records) if study_selection is not None else []
            study_jobs = [
                (study_id, study_records[index] if index < len(study_records) else None)
                for index, study_id in enumerate(study_ids)
            ]
        elif args.all_cancers:
            selections = study_selections or ()
            if len(selections) == 0:
                print("error: no studies found in frozen index", file=sys.stderr)
                return 1
            if args.output is not None:
                print("error: --output can only be used when a single study is selected", file=sys.stderr)
                return 1
            for selection in selections:
                for record in selection.study_records:
                    study_jobs.append((record.study.study_id, record))
        if len(study_jobs) > 1 and args.output is not None:
            print("error: --output can only be used when a single study is selected", file=sys.stderr)
            return 1

        for study_id, study_record in study_jobs:
            try:
                bundle = build_cbioportal_data_source(
                    client=client,
                    study_id=study_id,
                    mutation_genes=mutation_genes or None,
                    mrna_genes=mrna_genes,
                    sample_list_id=args.sample_list_id,
                    mutation_profile_id=study_record.mutation_profile_id if study_record is not None else None,
                    mrna_profile_id=study_record.mrna_profile_id if study_record is not None else None,
                    study=study_record.study if study_record is not None else None,
                )
                study = bundle.source.study
                output_dir = args.output_dir / study.study_id
                sample_list_id = bundle.source.sample_list_id
                sample_ids = list(bundle.source.sample_ids)
                mutation_table = bundle.mutation_table if bundle.mutation_table is not None else pd.DataFrame(index=[], columns=sample_ids)
                mrna_table = bundle.mrna_table if bundle.mrna_table is not None else pd.DataFrame(index=[], columns=sample_ids)
                mrna_raw_table = bundle.mrna_raw_table if bundle.mrna_raw_table is not None else mrna_table
                violin_plot_path = args.output if args.output is not None else (output_dir / "violin.png")
                forest_plot_path = output_dir / "forest.png"
                heatmap_plot_path = output_dir / "oncoplot.png"
                if args.figure_style == "relationship" and violin_plot_path == heatmap_plot_path:
                    heatmap_plot_path = output_dir / "heatmap.png"
                association_table = None
                has_mutation_signal = _has_any_mutation_signal(mutation_table)
                if args.figure_style == "relationship":
                    if not has_mutation_signal or mrna_raw_table.empty:
                        if args.all_cancers:
                            print(f"skip: {study.study_id} has no usable mutation signal or expression table")
                            continue
                        print("error: relationship figure requires both mutation and expression tables", file=sys.stderr)
                        return 1
                    association_table = compute_mutation_expression_associations(
                        mutation_table,
                        mrna_table,
                        mrna_raw_table=mrna_raw_table,
                        mutation_genes=mutation_genes,
                        expression_genes=mrna_genes,
                    )
                    if association_table.empty:
                        if args.all_cancers:
                            print(f"skip: {study.study_id} has no mutation-expression associations")
                            continue
                        print("error: no mutation-expression associations available to plot", file=sys.stderr)
                        return 1
                    plot_cbioportal_oncoplot(
                        output_path=heatmap_plot_path,
                        mutation_table=mutation_table,
                        mrna_table=mrna_table,
                        source=bundle.source,
                        title=f"{study.study_id} oncoplot",
                        presentation=True,
                        max_sample_labels=0,
                        hide_sample_labels=True,
                    )
                    forest_plot_paths = plot_mutation_expression_forest_summary(
                        output_path=forest_plot_path,
                        mutation_table=mutation_table,
                        mrna_table=mrna_table,
                        mrna_raw_table=mrna_raw_table,
                        source=bundle.source,
                        title=f"{study.study_id} mutation-expression summary",
                        association_table=association_table,
                        summary_label="Mutation-expression summary (log2(TPM))",
                        presentation=True,
                    )
                    plot_mutation_expression_violin_panels(
                        output_path=violin_plot_path,
                        mutation_table=mutation_table,
                        mrna_table=mrna_table,
                        mrna_raw_table=mrna_raw_table,
                        source=bundle.source,
                        title=f"{study.study_id} mutation-expression relationships",
                        association_table=association_table,
                        presentation=True,
                    )
                    plot_path = violin_plot_path
                else:
                    forest_plot_paths = None
                    plot_cbioportal_oncoplot(
                        output_path=heatmap_plot_path if args.output is None else args.output,
                        mutation_table=mutation_table,
                        mrna_table=mrna_table,
                        source=bundle.source,
                        title=f"{study.study_id} oncoplot",
                        presentation=args.presentation,
                        max_sample_labels=args.max_sample_labels,
                    )
                    plot_path = heatmap_plot_path if args.output is None else args.output
                result = save_cbioportal_outputs(
                    output_dir=output_dir,
                    study=study,
                    sample_list_id=sample_list_id,
                    sample_ids=sample_ids,
                    mutation_table=mutation_table,
                    mrna_table=mrna_table,
                    mrna_raw_table=mrna_raw_table,
                    output_plot=plot_path,
                    forest_plot_paths=forest_plot_paths if args.figure_style == "relationship" else None,
                    association_table=association_table,
                    source=bundle.source,
                )
                print(f"study: {result.study.study_id}")
                print(f"sample list: {result.sample_list_id}")
                print(f"wrote: {result.output_plot}")
                if args.figure_style == "relationship":
                    print(f"wrote: {heatmap_plot_path}")
                    for path in forest_plot_paths or []:
                        print(f"wrote: {path}")
                print(f"wrote: {result.mutation_table_path}")
                print(f"wrote: {result.mrna_table_path}")
                if result.mrna_raw_table_path is not None:
                    print(f"wrote: {result.mrna_raw_table_path}")
                print(f"wrote: {result.samples_path}")
                if result.association_table_path is not None:
                    print(f"wrote: {result.association_table_path}")
                if result.source_path is not None:
                    print(f"wrote: {result.source_path}")
            except ValueError as exc:
                if args.all_cancers and "no mutation-expression associations available to plot" in str(exc):
                    print(f"skip: {study_id} has no mutation-expression associations")
                    continue
                raise
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _has_any_mutation_signal(mutation_table: pd.DataFrame) -> bool:
    if mutation_table.empty:
        return False
    mutated = mutation_table.map(_is_mutated)
    return bool(mutated.to_numpy(dtype=bool).any())


if __name__ == "__main__":
    raise SystemExit(main())
