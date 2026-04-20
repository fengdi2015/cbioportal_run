from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geo_downloader.cbioportal import (
    CbioPortalClient,
    build_cbioportal_data_source,
    compute_mutation_expression_associations,
    plot_mutation_expression_forest_summary,
    plot_mutation_expression_violin_panels,
    plot_cbioportal_oncoplot,
    save_cbioportal_outputs,
)


DEFAULT_STUDY_ID = "luad_cptac_2020"
DEFAULT_MUTATION_GENES = ["STK11", "KRAS", "TP53"]
DEFAULT_EXPRESSION_GENES = ["CD3D", "MX1", "MX2"]
DEFAULT_MUTATION_PROFILE_ID = "luad_cptac_2020_mutations"
DEFAULT_EXPRESSION_PROFILE_ID = "luad_cptac_2020_rna_seq_mrna_median_all_sample_Zscores"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smoke_cbioportal_luad_cptac",
        description="Live smoke test for LUAD CPTAC 2020 with STK11/KRAS/TP53 and immune-response genes.",
    )
    parser.add_argument("--study-id", default=DEFAULT_STUDY_ID, help="cBioPortal study ID to query.")
    parser.add_argument(
        "--mutation-genes",
        default=",".join(DEFAULT_MUTATION_GENES),
        help="Comma-separated mutation genes; defaults to STK11,KRAS,TP53.",
    )
    parser.add_argument(
        "--expression-genes",
        default=",".join(DEFAULT_EXPRESSION_GENES),
        help="Comma-separated expression genes; defaults to CD3D,MX1,MX2.",
    )
    parser.add_argument(
        "--sample-list-id",
        help="Optional cBioPortal sample list ID. Defaults to the study's all-cases list.",
    )
    parser.add_argument(
        "--mutation-profile-id",
        default=DEFAULT_MUTATION_PROFILE_ID,
        help="Exact mutation profile ID to use.",
    )
    parser.add_argument(
        "--expression-profile-id",
        default=DEFAULT_EXPRESSION_PROFILE_ID,
        help="Exact mRNA Z-score profile ID to use.",
    )
    parser.add_argument(
        "--base-url",
        default="https://www.cbioportal.org",
        help="cBioPortal base URL, useful for private instances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "smoke-cbioportal-luad-cptac",
        help="Directory where the smoke-test outputs will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    mutation_genes = [item.strip() for item in args.mutation_genes.split(",") if item.strip()]
    expression_genes = [item.strip() for item in args.expression_genes.split(",") if item.strip()]
    if not mutation_genes:
        print("error: at least one mutation gene is required", file=sys.stderr)
        return 1
    if not expression_genes:
        print("error: at least one expression gene is required", file=sys.stderr)
        return 1

    client = CbioPortalClient(base_url=args.base_url)
    try:
        bundle = build_cbioportal_data_source(
            client=client,
            study_id=args.study_id,
            mutation_genes=mutation_genes,
            mrna_genes=expression_genes,
            sample_list_id=args.sample_list_id,
            mutation_profile_id=args.mutation_profile_id,
            mrna_profile_id=args.expression_profile_id,
        )
        study = bundle.source.study
        sample_list_id = bundle.source.sample_list_id
        sample_ids = list(bundle.source.sample_ids)
        mutation_table = bundle.mutation_table
        mrna_table = bundle.mrna_table
        mrna_raw_table = bundle.mrna_raw_table
        if mutation_table is None or mrna_table is None or mrna_raw_table is None:
            raise RuntimeError("expected both mutation and mRNA tables to be populated")
        if mutation_table.empty or mrna_table.empty or mrna_raw_table.empty:
            raise RuntimeError("expected both mutation and mRNA tables to be populated")

        output_dir = args.output_dir / study.study_id
        heatmap_path = output_dir / "oncoplot.png"
        violin_path = output_dir / "violin.png"
        forest_path = output_dir / "forest.png"
        plot_cbioportal_oncoplot(
            output_path=heatmap_path,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            source=bundle.source,
            title=f"{study.study_id} mutation and mRNA Z-score (n={len(sample_ids)})",
            expression_label="mRNA Z-score",
            presentation=True,
            max_sample_labels=0,
            hide_sample_labels=True,
        )
        association_table = compute_mutation_expression_associations(
            mutation_table,
            mrna_table,
            mrna_raw_table=mrna_raw_table,
            mutation_genes=mutation_genes,
            expression_genes=expression_genes,
        )
        plot_mutation_expression_violin_panels(
            output_path=violin_path,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            mrna_raw_table=mrna_raw_table,
            source=bundle.source,
            title=f"{study.study_id} mutation and mRNA Z-score (n={len(sample_ids)})",
            association_table=association_table,
            presentation=True,
        )
        forest_plot_paths = plot_mutation_expression_forest_summary(
            output_path=forest_path,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            mrna_raw_table=mrna_raw_table,
            source=bundle.source,
            title=f"{study.study_id} mutation and mRNA Z-score (n={len(sample_ids)})",
            association_table=association_table,
            summary_label="LUAD CPTAC mutation-expression summary (log2(TPM))",
            presentation=True,
        )
        result = save_cbioportal_outputs(
            output_dir=output_dir,
            study=study,
            sample_list_id=sample_list_id,
            sample_ids=sample_ids,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            mrna_raw_table=mrna_raw_table,
            output_plot=violin_path,
            forest_plot_paths=forest_plot_paths,
            association_table=association_table,
            source=bundle.source,
        )

        print(f"study: {result.study.study_id}")
        print(f"sample list: {result.sample_list_id}")
        print(f"samples: {len(result.sample_ids)}")
        print(f"wrote: {result.output_plot}")
        for path in forest_plot_paths:
            print(f"wrote: {path}")
        print(f"wrote: {heatmap_path}")
        print(f"wrote: {result.mutation_table_path}")
        print(f"wrote: {result.mrna_table_path}")
        if result.mrna_raw_table_path is not None:
            print(f"wrote: {result.mrna_raw_table_path}")
        print(f"wrote: {result.samples_path}")
        if result.association_table_path is not None:
            print(f"wrote: {result.association_table_path}")
        if result.source_path is not None:
            print(f"wrote: {result.source_path}")
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
