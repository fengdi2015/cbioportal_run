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


DEFAULT_STUDY_ID = "coadread_tcga_pan_can_atlas_2018"
DEFAULT_GENES = ["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "FBXW7"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smoke_cbioportal_coad",
        description="Live smoke test for cBioPortal COAD data and oncoplot generation.",
    )
    parser.add_argument("--study-id", default=DEFAULT_STUDY_ID, help="cBioPortal study ID to query.")
    parser.add_argument(
        "--genes",
        default=",".join(DEFAULT_GENES),
        help="Comma-separated driver genes used for both mutation and mRNA rows.",
    )
    parser.add_argument(
        "--sample-list-id",
        help="Optional cBioPortal sample list ID. Defaults to the study's all-cases list.",
    )
    parser.add_argument(
        "--base-url",
        default="https://www.cbioportal.org",
        help="cBioPortal base URL, useful for private instances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads") / "smoke-cbioportal-coad",
        help="Directory where the smoke-test outputs will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    genes = [item.strip() for item in args.genes.split(",") if item.strip()]
    if not genes:
        print("error: at least one gene is required", file=sys.stderr)
        return 1

    client = CbioPortalClient(base_url=args.base_url)
    try:
        bundle = build_cbioportal_data_source(
            client=client,
            study_id=args.study_id,
            mutation_genes=genes,
            mrna_genes=genes,
            sample_list_id=args.sample_list_id,
        )
        study = bundle.source.study
        sample_list_id = bundle.source.sample_list_id
        sample_ids = list(bundle.source.sample_ids)
        mutation_table = bundle.mutation_table
        mrna_table = bundle.mrna_table
        mrna_raw_table = bundle.mrna_raw_table

        if not sample_ids:
            raise RuntimeError("no samples were returned from cBioPortal")
        if mutation_table is None or mrna_table is None or mrna_raw_table is None:
            raise RuntimeError("expected both mutation and mRNA tables to be populated")
        if mutation_table.empty or mrna_table.empty or mrna_raw_table.empty:
            raise RuntimeError("expected both mutation and mRNA tables to be populated")
        if not mutation_table.replace("", None).stack().dropna().any():
            raise RuntimeError("mutation table contains no observed alterations")
        if not mrna_raw_table.stack().dropna().size:
            raise RuntimeError("mRNA table contains no finite values")

        output_dir = args.output_dir / study.study_id
        heatmap_path = output_dir / "oncoplot.png"
        violin_path = output_dir / "violin.png"
        forest_path = output_dir / "forest.png"
        plot_cbioportal_oncoplot(
            output_path=heatmap_path,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            source=bundle.source,
            title=f"{study.study_id} smoke test",
            presentation=True,
            max_sample_labels=0,
            hide_sample_labels=True,
        )
        association_table = compute_mutation_expression_associations(
            mutation_table,
            mrna_table,
            mrna_raw_table=mrna_raw_table,
        )
        plot_mutation_expression_violin_panels(
            output_path=violin_path,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            mrna_raw_table=mrna_raw_table,
            source=bundle.source,
            title=f"{study.study_id} mutation-expression relationships",
            association_table=association_table,
            presentation=True,
        )
        forest_plot_paths = plot_mutation_expression_forest_summary(
            output_path=forest_path,
            mutation_table=mutation_table,
            mrna_table=mrna_table,
            mrna_raw_table=mrna_raw_table,
            source=bundle.source,
            title=f"{study.study_id} mutation-expression summary",
            association_table=association_table,
            summary_label="COAD mutation-expression summary (log2(TPM))",
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
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
