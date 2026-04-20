import tarfile
import io
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from geo_downloader.cbioportal import (
    CbioPortalDataBundle,
    CbioPortalDataSource,
    CbioPortalClient,
    CbioPortalProfile,
    CbioPortalStudy,
    CbioPortalStudyReference,
    CbioPortalStudySelection,
    build_cbioportal_study_index,
    build_cbioportal_data_source,
    compute_mutation_expression_associations,
    build_cbioportal_matrices,
    load_cbioportal_study_index,
    _format_gene_pair_label,
    plot_mutation_expression_forest_summary,
    plot_mutation_expression_violin_panels,
    plot_cbioportal_oncoplot,
    save_cbioportal_outputs,
    save_cbioportal_study_index,
)
from geo_downloader.cbioportal_cli import main as cbioportal_main
from geo_downloader.cbioportal_index_cli import main as cbioportal_index_main
from geo_downloader.cli import _write_graph
from geo_downloader.geo import build_geo_record, geo_series_download_url
from geo_downloader.h5ad import convert_all_sample_bundles_to_h5ad, convert_sample_bundle_to_h5ad, list_sample_bundles
from geo_downloader.workflow import GeoDownloadWorkflow, WorkflowSettings
from scripts.smoke_cbioportal_coad import main as smoke_cbioportal_coad_main
from scripts.smoke_cbioportal_luad import main as smoke_cbioportal_luad_main
from scripts.smoke_cbioportal_luad_cptac import main as smoke_cbioportal_luad_cptac_main


class FakeCbioPortalClient(CbioPortalClient):
    def __init__(self) -> None:
        pass

    def get_study(self, study_id: str) -> CbioPortalStudy:
        names = {
            "luad_tcga_gdc": "Lung Adenocarcinoma (TCGA GDC, 2025)",
            "luad_cptac_2020": "Lung Adenocarcinoma (CPTAC, Cell 2020)",
            "luad_cptac_gdc": "Lung Adenocarcinoma (CPTAC GDC, 2025)",
            "paad_tcga_gdc": "Pancreatic Adenocarcinoma (TCGA GDC, 2025)",
            "paad_cptac_2021": "Pancreatic Ductal Adenocarcinoma (CPTAC, Cell 2021)",
            "brca_tcga_gdc": "Invasive Breast Carcinoma (TCGA GDC, 2025)",
            "brca_cptac_2020": "Proteogenomic landscape of breast cancer (CPTAC, Cell 2020)",
            "stad_tcga_gdc": "Stomach Adenocarcinoma (TCGA GDC, 2025)",
        }
        return CbioPortalStudy(study_id=study_id, name=names.get(study_id, "Toy Study"))

    def get_sample_lists(self, study_id: str):
        return [{"sampleListId": f"{study_id}_all", "category": "all_cases_in_study"}]

    def get_studies(self):
        return [
            {"studyId": "luad_tcga_gdc", "name": "Lung Adenocarcinoma (TCGA GDC, 2025)"},
            {"studyId": "luad_cptac_2020", "name": "Lung Adenocarcinoma (CPTAC, Cell 2020)"},
            {"studyId": "luad_cptac_gdc", "name": "Lung Adenocarcinoma (CPTAC GDC, 2025)"},
            {"studyId": "paad_tcga_gdc", "name": "Pancreatic Adenocarcinoma (TCGA GDC, 2025)"},
            {"studyId": "paad_cptac_2021", "name": "Pancreatic Ductal Adenocarcinoma (CPTAC, Cell 2021)"},
            {"studyId": "brca_tcga_gdc", "name": "Invasive Breast Carcinoma (TCGA GDC, 2025)"},
            {"studyId": "brca_cptac_2020", "name": "Proteogenomic landscape of breast cancer (CPTAC, Cell 2020)"},
            {"studyId": "stad_tcga_gdc", "name": "Stomach Adenocarcinoma (TCGA GDC, 2025)"},
        ]

    def get_sample_list(self, sample_list_id: str):
        return ["S1", "S2", "S3"]

    def get_molecular_profiles(self, study_id: str):
        return [
            CbioPortalProfile("toy_mutations", study_id, alteration_type="MUTATION_EXTENDED", datatype="MAF"),
            CbioPortalProfile("toy_mrna", study_id, alteration_type="MRNA_EXPRESSION", datatype="CONTINUOUS"),
        ]

    def resolve_gene(self, gene: str):
        mapping = {
            "STK11": 11200,
            "APC": 324,
            "TP53": 7157,
            "KRAS": 3845,
            "PIK3CA": 5290,
            "SMAD4": 4089,
            "FBXW7": 55294,
            "MYC": 4609,
            "CD3D": 915,
            "MX1": 4599,
            "MX2": 4600,
            "CD3E": 916,
            "CD247": 919,
            "TRAC": 6973,
            "LCK": 3932,
            "IL7R": 3575,
            "CCR7": 1236,
            "PTPRC": 5788,
            "CD68": 968,
            "CD163": 9332,
            "CSF1R": 1436,
            "LYZ": 3934,
            "C1QA": 712,
            "C1QB": 713,
            "C1QC": 714,
            "MSR1": 4481,
            "COL1A1": 1277,
            "COL1A2": 1278,
            "ACTA2": 59,
            "PDGFRB": 5159,
            "PECAM1": 5175,
            "VWF": 7450,
            "CXCL9": 4283,
            "CXCL10": 3627,
            "CXCL11": 6373,
            "TGFB1": 7040,
            "ITGAM": 3684,
            "ITGB2": 3689,
        }
        return {"entrezGeneId": mapping[gene], "hugoGeneSymbol": gene}

    def fetch_mutations(self, molecular_profile_id: str, sample_list_id: str, entrez_gene_id: int):
        import pandas as pd

        if entrez_gene_id == 324:
            return pd.DataFrame([{"sampleId": "S1", "mutationType": "Frame_Shift_Del"}])
        if entrez_gene_id == 11200:
            return pd.DataFrame(
                [
                    {"sampleId": "S1", "mutationType": "Missense_Mutation"},
                    {"sampleId": "S4", "mutationType": "Splice_Site"},
                ]
            )
        if entrez_gene_id == 7157:
            return pd.DataFrame(
                [
                    {"sampleId": "S1", "mutationType": "Missense_Mutation"},
                    {"sampleId": "S3", "mutationType": "Nonsense_Mutation"},
                ]
            )
        if entrez_gene_id == 3845:
            return pd.DataFrame([{"sampleId": "S2", "mutationType": "Missense_Mutation"}])
        if entrez_gene_id == 5290:
            return pd.DataFrame([{"sampleId": "S3", "mutationType": "Missense_Mutation"}])
        if entrez_gene_id == 4089:
            return pd.DataFrame([{"sampleId": "S2", "mutationType": "In_Frame_Ins"}])
        if entrez_gene_id == 55294:
            return pd.DataFrame([{"sampleId": "S1", "mutationType": "Nonsense_Mutation"}])
        return pd.DataFrame([])

    def fetch_expression(self, molecular_profile_id: str, sample_list_id: str, entrez_gene_id: int):
        import pandas as pd

        values = {
            11200: [-1.4, -0.8, -0.9, -1.2],
            324: [1.5, 0.1, -0.2, 0.3],
            7157: [1.2, -0.3, 0.5, 0.7],
            3845: [0.8, -0.5, 0.2, -0.1],
            5290: [-0.7, 0.9, 1.1, 0.8],
            4089: [0.4, -1.4, 0.6, -0.2],
            55294: [1.0, 0.0, -0.8, 0.2],
            4609: [0.2, 0.6, -1.1, -0.4],
            915: [2.1, 1.9, 2.0, 1.8],
            4599: [1.6, 1.5, 1.7, 1.4],
            4600: [1.8, 1.7, 1.9, 1.6],
            916: [2.0, 2.1, 2.2, 1.9],
            919: [1.8, 2.0, 1.9, 1.7],
            6973: [1.4, 1.3, 1.5, 1.2],
            3932: [0.9, 1.0, 0.8, 0.7],
            3575: [1.5, 1.4, 1.6, 1.3],
            1236: [1.1, 1.0, 1.2, 0.9],
            5788: [2.2, 2.1, 2.3, 2.0],
            968: [0.2, 0.1, 0.0, 0.3],
            9332: [0.4, 0.2, 0.5, 0.3],
            1436: [0.6, 0.7, 0.5, 0.8],
            3934: [0.3, 0.2, 0.4, 0.1],
            712: [0.9, 0.8, 1.0, 0.7],
            713: [1.0, 0.9, 1.1, 0.8],
            714: [0.8, 0.7, 0.9, 0.6],
            4481: [0.5, 0.4, 0.6, 0.3],
            1277: [2.5, 2.6, 2.4, 2.7],
            1278: [2.4, 2.3, 2.5, 2.6],
            59: [1.7, 1.6, 1.8, 1.5],
            5159: [0.4, 0.3, 0.5, 0.2],
            5175: [1.1, 1.0, 1.2, 0.9],
            7450: [0.6, 0.5, 0.7, 0.4],
            4283: [1.9, 2.0, 1.8, 1.7],
            3627: [2.1, 2.2, 2.0, 1.9],
            6373: [2.0, 2.1, 1.9, 1.8],
            7040: [1.4, 1.5, 1.3, 1.2],
            3684: [0.8, 0.7, 0.9, 0.6],
            3689: [0.9, 0.8, 1.0, 0.7],
        }
        if entrez_gene_id not in values:
            return pd.DataFrame([])
        sample_ids = ["S1", "S2", "S3", "S4"]
        return pd.DataFrame({"sampleId": sample_ids, "value": values[entrez_gene_id]})


def test_geo_download_url_for_series():
    assert geo_series_download_url("GSE150290") == (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE150290&format=file"
    )


def test_build_geo_record_ignores_geo_directories():
    html = """
    <html>
      <body>
        <a href="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE150nnn/GSE150290/soft/">soft</a>
        <a href="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE150nnn/GSE150290/matrix/">matrix</a>
        <a href="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE150nnn/GSE150290/GSE150290_RAW.tar">raw</a>
      </body>
    </html>
    """
    record = build_geo_record("GSE150290", html)
    assert len(record.artifacts) == 1
    assert record.artifacts[0].filename == "GSE150290_RAW.tar"


def test_workflow_writes_manifest_and_report(tmp_path, monkeypatch):
    html = """
    <html>
      <body>
        <a href="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE150nnn/GSE150290/GSE150290_RAW.tar">raw</a>
        <div>SRP261119</div>
      </body>
    </html>
    """

    def fake_fetch_text(_url: str, timeout: int = 30) -> str:
        return html

    def fake_download_file(_url: str, destination: Path, timeout: int = 60) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("data", encoding="utf-8")
        return 4

    monkeypatch.setattr("geo_downloader.workflow.fetch_text", fake_fetch_text)
    monkeypatch.setattr("geo_downloader.workflow.download_file", fake_download_file)

    workflow = GeoDownloadWorkflow(WorkflowSettings(no_progress=True, no_summaries=True))
    result = workflow.run("GSE150290", tmp_path)

    assert result.manifest_path.exists()
    assert result.report_path.exists()
    assert (tmp_path / "GSE150290" / "files" / "GSE150290_RAW.tar").exists()


def test_write_graph(tmp_path):
    graph_path = tmp_path / "graph.mmd"
    _write_graph(graph_path)
    assert graph_path.exists()


def test_convert_sample_bundle_to_h5ad(tmp_path):
    inner_bytes = io.BytesIO()
    with tarfile.open(fileobj=inner_bytes, mode="w:gz") as inner:
        matrix = csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32))
        matrix_buf = io.BytesIO()
        mmwrite(matrix_buf, matrix)
        _add_bytes(inner, "raw_gene_bc_matrices/hg19/matrix.mtx", matrix_buf.getvalue())
        _add_text(inner, "raw_gene_bc_matrices/hg19/genes.tsv", "ENSG1\tGENE1\nENSG2\tGENE2\n")
        _add_text(inner, "raw_gene_bc_matrices/hg19/barcodes.tsv", "BC1\nBC2\nBC3\n")

    outer_path = tmp_path / "GSE150290_RAW.tar"
    with tarfile.open(outer_path, mode="w") as outer:
        info = tarfile.TarInfo("GSM000000_Test.raw_gene_bc_matrices.tar.gz")
        payload = inner_bytes.getvalue()
        info.size = len(payload)
        outer.addfile(info, io.BytesIO(payload))

    bundles = list_sample_bundles(outer_path)
    assert [bundle.sample_id for bundle in bundles] == ["GSM000000_Test"]

    out = tmp_path / "GSM000000_Test.h5ad"
    convert_sample_bundle_to_h5ad(outer_path, "GSM000000_Test", out)
    adata = ad.read_h5ad(out)

    assert adata.shape == (3, 2)
    assert list(adata.obs["sample_id"].unique()) == ["GSM000000_Test"]
    assert list(adata.var["gene_name"]) == ["GENE1", "GENE2"]


def test_convert_sample_bundle_handles_gzipped_inner_files(tmp_path):
    inner_bytes = io.BytesIO()
    with tarfile.open(fileobj=inner_bytes, mode="w:gz") as inner:
        matrix = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
        matrix_buf = io.BytesIO()
        mmwrite(matrix_buf, matrix)
        _add_bytes(inner, "raw_gene_bc_matrices/hg19/matrix.mtx.gz", gzip.compress(matrix_buf.getvalue()))
        _add_bytes(
            inner,
            "raw_gene_bc_matrices/hg19/features.tsv.gz",
            gzip.compress(b"ENSG1\tGENE1\tGene Expression\nENSG2\tGENE2\tGene Expression\n"),
        )
        _add_bytes(inner, "raw_gene_bc_matrices/hg19/barcodes.tsv.gz", gzip.compress(b"BC1\nBC2\n"))

    outer_path = tmp_path / "GSE150290_RAW.tar"
    with tarfile.open(outer_path, mode="w") as outer:
        info = tarfile.TarInfo("GSM000001_Test.raw_gene_bc_matrices.tar.gz")
        payload = inner_bytes.getvalue()
        info.size = len(payload)
        outer.addfile(info, io.BytesIO(payload))

    out = tmp_path / "GSM000001_Test.h5ad"
    convert_sample_bundle_to_h5ad(outer_path, "GSM000001_Test", out)
    adata = ad.read_h5ad(out)

    assert adata.shape == (2, 2)
    assert list(adata.var["feature_type"]) == ["Gene Expression", "Gene Expression"]


def test_convert_all_sample_bundles_to_h5ad(tmp_path):
    outer_path = tmp_path / "GSE150290_RAW.tar"
    _write_multi_sample_archive(outer_path, ["GSM000010_TestA", "GSM000011_TestB"])

    out_dir = tmp_path / "h5ads"
    written = convert_all_sample_bundles_to_h5ad(outer_path, out_dir)

    assert sorted(path.name for path in written) == ["GSM000010_TestA.h5ad", "GSM000011_TestB.h5ad"]
    assert all(path.exists() for path in written)


def test_cbioportal_matrices_and_plot(tmp_path):
    client = FakeCbioPortalClient()
    bundle = build_cbioportal_data_source(
        client=client,
        study_id="toy_study",
        mutation_genes=["TP53", "KRAS"],
        mrna_genes=["TP53", "MYC"],
    )
    study, sample_list_id, sample_ids, mutation_table, mrna_table = build_cbioportal_matrices(
        client=client,
        study_id="toy_study",
        mutation_genes=["TP53", "KRAS"],
        mrna_genes=["TP53", "MYC"],
    )

    assert study.study_id == "toy_study"
    assert sample_list_id == "toy_study_all"
    assert sample_ids == ["S1", "S2", "S3"]
    assert list(mutation_table.index) == ["TP53", "KRAS"]
    assert list(mrna_table.index) == ["TP53", "MYC"]
    assert mutation_table.loc["TP53", "S1"] == "Missense_Mutation"
    assert mutation_table.loc["KRAS", "S2"] == "Missense_Mutation"
    assert abs(float(pd.to_numeric(mrna_table.loc["MYC"], errors="coerce").mean())) < 1e-12
    assert abs(float(pd.to_numeric(mrna_table.loc["MYC"], errors="coerce").std(ddof=0)) - 1.0) < 1e-12
    assert bundle.mrna_raw_table is not None
    assert bundle.mrna_raw_table.loc["MYC", "S3"] == -1.1

    output_dir = tmp_path / "cbio"
    plot_path = output_dir / "oncoplot.png"
    result = save_cbioportal_outputs(
        output_dir=output_dir,
        study=study,
        sample_list_id=sample_list_id,
        sample_ids=sample_ids,
        mutation_table=mutation_table,
        mrna_table=mrna_table,
        output_plot=plot_path,
    )
    plot_cbioportal_oncoplot(
        plot_path,
        mutation_table=mutation_table,
        mrna_table=mrna_table,
        source=bundle.source,
        title="toy_study oncoplot",
    )

    assert result.mutation_table_path.exists()
    assert result.mrna_table_path.exists()
    assert result.samples_path.exists()
    assert plot_path.exists()


def test_cbioportal_relationship_plot(tmp_path):
    client = FakeCbioPortalClient()
    bundle = build_cbioportal_data_source(
        client=client,
        study_id="toy_study",
        mutation_genes=["TP53", "KRAS"],
        mrna_genes=["TP53", "MYC"],
    )
    association = compute_mutation_expression_associations(
        bundle.mutation_table,
        bundle.mrna_table,
        mrna_raw_table=bundle.mrna_raw_table,
        mutation_genes=["TP53", "KRAS"],
        expression_genes=["TP53", "MYC"],
    )
    violin_path = tmp_path / "violin.png"
    forest_path = tmp_path / "forest.png"
    plot_mutation_expression_violin_panels(
        violin_path,
        mutation_table=bundle.mutation_table,
        mrna_table=bundle.mrna_table,
        mrna_raw_table=bundle.mrna_raw_table,
        source=bundle.source,
        title="toy relationship",
        association_table=association,
    )
    forest_paths = plot_mutation_expression_forest_summary(
        forest_path,
        mutation_table=bundle.mutation_table,
        mrna_table=bundle.mrna_table,
        mrna_raw_table=bundle.mrna_raw_table,
        source=bundle.source,
        title="toy relationship",
        association_table=association,
        summary_label="toy summary",
    )
    assert violin_path.exists()
    assert sorted(path.name for path in forest_paths) == ["forest_KRAS.png", "forest_TP53.png"]
    assert all(path.exists() for path in forest_paths)


def test_cbioportal_gene_pair_label_formats_both_genes():
    assert _format_gene_pair_label("STK11", "CD68") == "Mut: STK11 | Expr: CD68"
    assert _format_gene_pair_label("TP53", "TP53") == "Gene: TP53"


def test_cbioportal_forest_summary_marks_significant_pvalues(tmp_path):
    client = FakeCbioPortalClient()
    bundle = build_cbioportal_data_source(
        client=client,
        study_id="toy_study",
        mutation_genes=["TP53", "KRAS"],
        mrna_genes=["TP53", "MYC"],
    )
    association = compute_mutation_expression_associations(
        bundle.mutation_table,
        bundle.mrna_table,
        mrna_raw_table=bundle.mrna_raw_table,
        mutation_genes=["TP53", "KRAS"],
        expression_genes=["TP53", "MYC"],
    )
    forest_path = tmp_path / "forest.png"
    forest_paths = plot_mutation_expression_forest_summary(
        forest_path,
        mutation_table=bundle.mutation_table,
        mrna_table=bundle.mrna_table,
        mrna_raw_table=bundle.mrna_raw_table,
        source=bundle.source,
        association_table=association,
        summary_label="toy summary",
    )
    assert sorted(path.name for path in forest_paths) == ["forest_KRAS.png", "forest_TP53.png"]
    assert all(path.exists() for path in forest_paths)


def test_cbioportal_coad_driver_mutations_and_expression(tmp_path):
    client = FakeCbioPortalClient()
    bundle = build_cbioportal_data_source(
        client=client,
        study_id="coadread_tcga_pan_can_atlas_2018",
        mutation_genes=["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "FBXW7"],
        mrna_genes=["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "FBXW7"],
    )
    study, sample_list_id, sample_ids, mutation_table, mrna_table = build_cbioportal_matrices(
        client=client,
        study_id="coadread_tcga_pan_can_atlas_2018",
        mutation_genes=["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "FBXW7"],
        mrna_genes=["APC", "TP53", "KRAS", "PIK3CA", "SMAD4", "FBXW7"],
    )

    assert study.study_id == "coadread_tcga_pan_can_atlas_2018"
    assert sample_list_id == "coadread_tcga_pan_can_atlas_2018_all"
    assert sample_ids == ["S1", "S2", "S3"]
    assert mutation_table.loc["APC", "S1"] == "Frame_Shift_Del"
    assert mutation_table.loc["TP53", "S3"] == "Nonsense_Mutation"
    assert mutation_table.loc["KRAS", "S2"] == "Missense_Mutation"
    pik3ca = pd.to_numeric(mrna_table.loc["PIK3CA"], errors="coerce")
    smad4 = pd.to_numeric(mrna_table.loc["SMAD4"], errors="coerce")
    assert abs(float(pik3ca.mean())) < 1e-12
    assert abs(float(pik3ca.std(ddof=0)) - 1.0) < 1e-12
    assert abs(float(smad4.mean())) < 1e-12
    assert abs(float(smad4.std(ddof=0)) - 1.0) < 1e-12

    plot_path = tmp_path / "coad" / "oncoplot.png"
    plot_cbioportal_oncoplot(
        plot_path,
        mutation_table=mutation_table,
        mrna_table=mrna_table,
        source=bundle.source,
        title="COAD driver mutations and expression",
        presentation=True,
    )
    association = compute_mutation_expression_associations(mutation_table, mrna_table)
    assert plot_path.exists()
    assert not association.empty
    assert {"mutation_gene", "expression_gene", "delta_median", "p_value"}.issubset(set(association.columns))


def test_cbioportal_source_manifest_and_single_modality_plot(tmp_path):
    client = FakeCbioPortalClient()
    bundle = build_cbioportal_data_source(
        client=client,
        study_id="toy_study",
        mutation_genes=["TP53", "KRAS"],
        mrna_genes=None,
    )

    assert bundle.source.mutation_profile is not None
    assert bundle.source.mrna_profile is not None
    assert bundle.mutation_table is not None
    assert bundle.mrna_table is None

    plot_path = tmp_path / "mutation_only.png"
    plot_cbioportal_oncoplot(
        plot_path,
        mutation_table=bundle.mutation_table,
        source=bundle.source,
        title="mutation only",
        presentation=True,
    )
    result = save_cbioportal_outputs(
        output_dir=tmp_path / "out",
        study=bundle.source.study,
        sample_list_id=bundle.source.sample_list_id,
        sample_ids=list(bundle.source.sample_ids),
        mutation_table=bundle.mutation_table,
        mrna_table=pd.DataFrame(index=[], columns=bundle.source.sample_ids),
        output_plot=plot_path,
        source=bundle.source,
    )

    assert plot_path.exists()
    assert result.source_path is not None and result.source_path.exists()


def test_cbioportal_tcga_expression_is_zscored():
    client = FakeCbioPortalClient()
    bundle = build_cbioportal_data_source(
        client=client,
        study_id="luad_tcga_gdc",
        mutation_genes=["TP53"],
        mrna_genes=["TP53"],
    )

    assert bundle.source.mrna_transform == "zscore"
    values = pd.to_numeric(bundle.mrna_table.loc["TP53"], errors="coerce")
    assert abs(float(values.mean())) < 1e-12
    assert abs(float(values.std(ddof=0)) - 1.0) < 1e-12


def test_resolve_default_studies_for_cancer():
    client = FakeCbioPortalClient()
    selection = client.resolve_default_studies_for_cancer("luad")

    assert selection.study_ids == ("luad_tcga_gdc", "luad_cptac_2020")
    assert selection.study_records[0].sample_list_id == "luad_tcga_gdc_all"


def test_build_and_load_cbioportal_study_index(tmp_path):
    client = FakeCbioPortalClient()
    index = build_cbioportal_study_index(client)
    output = save_cbioportal_study_index(index, tmp_path / "cbioportal_study_index.json")

    loaded = load_cbioportal_study_index(output)
    assert output.exists()
    assert loaded.entries[0].cancer_id == "brca"
    assert any(entry.cancer_id == "luad" for entry in loaded.entries)
    luad_entry = next(entry for entry in loaded.entries if entry.cancer_id == "luad")
    assert [record.study.study_id for record in luad_entry.study_records] == ["luad_tcga_gdc", "luad_cptac_2020"]
    assert luad_entry.study_records[0].mutation_profile_id == "toy_mutations"
    assert luad_entry.study_records[0].mrna_profile_id == "toy_mrna"


def test_cbioportal_accepts_gene_iterables():
    client = FakeCbioPortalClient()
    mutation_genes = (gene for gene in ["TP53", "KRAS"])
    mrna_genes = (gene for gene in ["TP53", "MYC"])
    _, _, _, mutation_table, mrna_table = build_cbioportal_matrices(
        client=client,
        study_id="toy_study",
        mutation_genes=mutation_genes,
        mrna_genes=mrna_genes,
    )

    assert list(mutation_table.index) == ["TP53", "KRAS"]
    assert list(mrna_table.index) == ["TP53", "MYC"]


def test_cbioportal_cli_smoke(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("geo_downloader.cbioportal_cli.CbioPortalClient", lambda base_url: client)
    rc = cbioportal_main(
        [
            "--study-id",
            "toy_study",
            "--mutation-genes",
            "TP53,KRAS",
            "--mrna-genes",
            "TP53,MYC",
            "--presentation",
            "--max-sample-labels",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
            "--output",
            str(tmp_path / "out" / "toy_study" / "violin.png"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "toy_study" / "violin.png").exists()
    assert (tmp_path / "out" / "toy_study" / "forest_KRAS.png").exists()
    assert (tmp_path / "out" / "toy_study" / "forest_TP53.png").exists()
    assert (tmp_path / "out" / "toy_study" / "oncoplot.png").exists()
    assert (tmp_path / "out" / "toy_study" / "mrna_raw.tsv").exists()


def test_cbioportal_cli_relationship_smoke(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("geo_downloader.cbioportal_cli.CbioPortalClient", lambda base_url: client)
    rc = cbioportal_main(
        [
            "--study-id",
            "toy_study",
            "--mutation-genes",
            "TP53,KRAS",
            "--mrna-genes",
            "TP53,MYC",
            "--figure-style",
            "relationship",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "toy_study" / "violin.png").exists()
    assert (tmp_path / "out" / "toy_study" / "forest_KRAS.png").exists()
    assert (tmp_path / "out" / "toy_study" / "forest_TP53.png").exists()
    assert (tmp_path / "out" / "toy_study" / "oncoplot.png").exists()
    assert (tmp_path / "out" / "toy_study" / "association.tsv").exists()
    assert (tmp_path / "out" / "toy_study" / "mrna_raw.tsv").exists()


def test_cbioportal_cli_uses_index_file_for_cancer_mode(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("geo_downloader.cbioportal_index_cli.CbioPortalClient", lambda base_url: client)
    index_path = tmp_path / "index.json"
    rc = cbioportal_index_main(["--output", str(index_path)])
    assert rc == 0
    assert index_path.exists()

    monkeypatch.setattr("geo_downloader.cbioportal_cli.CbioPortalClient", lambda base_url: client)
    rc = cbioportal_main(
        [
            "--cancer",
            "luad",
            "--mutation-genes",
            "TP53,KRAS",
            "--mrna-genes",
            "TP53,MYC",
            "--index-file",
            str(index_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "luad_tcga_gdc" / "oncoplot.png").exists()
    assert (tmp_path / "out" / "luad_cptac_2020" / "oncoplot.png").exists()
    assert (tmp_path / "out" / "luad_tcga_gdc" / "violin.png").exists()
    assert (tmp_path / "out" / "luad_cptac_2020" / "violin.png").exists()
    assert (tmp_path / "out" / "luad_tcga_gdc" / "forest_KRAS.png").exists()
    assert (tmp_path / "out" / "luad_cptac_2020" / "forest_KRAS.png").exists()
    assert (tmp_path / "out" / "luad_tcga_gdc" / "forest_TP53.png").exists()
    assert (tmp_path / "out" / "luad_cptac_2020" / "forest_TP53.png").exists()
    assert (tmp_path / "out" / "luad_tcga_gdc" / "mrna_raw.tsv").exists()
    assert (tmp_path / "out" / "luad_cptac_2020" / "mrna_raw.tsv").exists()


def test_cbioportal_cli_all_cancers_mode(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("geo_downloader.cbioportal_index_cli.CbioPortalClient", lambda base_url: client)
    index_path = tmp_path / "index.json"
    rc = cbioportal_index_main(["--output", str(index_path)])
    assert rc == 0
    assert index_path.exists()

    monkeypatch.setattr("geo_downloader.cbioportal_cli.CbioPortalClient", lambda base_url: client)
    rc = cbioportal_main(
        [
            "--all-cancers",
            "--mutation-genes",
            "TP53,KRAS",
            "--mrna-genes",
            "TP53,MYC",
            "--index-file",
            str(index_path),
            "--output-dir",
            str(tmp_path / "all"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "all" / "luad_tcga_gdc" / "violin.png").exists()
    assert (tmp_path / "all" / "luad_tcga_gdc" / "forest_TP53.png").exists()
    assert (tmp_path / "all" / "luad_cptac_2020" / "forest_KRAS.png").exists()
    assert (tmp_path / "all" / "brca_tcga_gdc" / "violin.png").exists()


def test_cbioportal_cli_all_cancers_skips_no_mutation_signal(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index_path.write_text("{}", encoding="utf-8")

    skip_study = CbioPortalStudy(study_id="skip_tcga_gdc", name="Skip Cancer (TCGA GDC, 2025)")
    keep_study = CbioPortalStudy(study_id="keep_tcga_gdc", name="Keep Cancer (TCGA GDC, 2025)")
    skip_record = CbioPortalStudyReference(
        role="tcga_gdc_2025",
        study=skip_study,
        sample_list_id="skip_tcga_gdc_all",
        sample_count=2,
        mutation_profile_id="skip_mutations",
        mrna_profile_id="skip_mrna",
    )
    keep_record = CbioPortalStudyReference(
        role="tcga_gdc_2025",
        study=keep_study,
        sample_list_id="keep_tcga_gdc_all",
        sample_count=2,
        mutation_profile_id="keep_mutations",
        mrna_profile_id="keep_mrna",
    )
    skip_selection = CbioPortalStudySelection(
        cancer_query="skip",
        study_ids=("skip_tcga_gdc",),
        studies=(skip_study,),
        study_records=(skip_record,),
    )
    keep_selection = CbioPortalStudySelection(
        cancer_query="keep",
        study_ids=("keep_tcga_gdc",),
        studies=(keep_study,),
        study_records=(keep_record,),
    )

    def fake_resolve_all_studies_from_index(_index):
        return (skip_selection, keep_selection)

    def fake_build_cbioportal_data_source(*, study_id: str, **_kwargs):
        if study_id == "skip_tcga_gdc":
            source = CbioPortalDataSource(
                study=skip_study,
                sample_list_id="skip_tcga_gdc_all",
                sample_ids=("S1", "S2"),
                mutation_profile=CbioPortalProfile("skip_mutations", "skip_tcga_gdc", "MUTATION_EXTENDED", None, "Mutations"),
                mrna_profile=CbioPortalProfile("skip_mrna", "skip_tcga_gdc", "MRNA_EXPRESSION", "RNA-Seq", "mRNA"),
                mrna_raw_profile=CbioPortalProfile("skip_mrna_raw", "skip_tcga_gdc", "MRNA_EXPRESSION", "RNA-Seq", "mRNA raw"),
                mrna_transform=None,
                mutation_genes=("TP53",),
                mrna_genes=("MYC",),
            )
            mutation_table = pd.DataFrame([["", ""]], index=["TP53"], columns=["S1", "S2"])
            mrna_table = pd.DataFrame([[0.0, 0.0]], index=["MYC"], columns=["S1", "S2"])
            return CbioPortalDataBundle(source=source, mutation_table=mutation_table, mrna_table=mrna_table, mrna_raw_table=mrna_table)
        source = CbioPortalDataSource(
            study=keep_study,
            sample_list_id="keep_tcga_gdc_all",
            sample_ids=("S1", "S2"),
            mutation_profile=CbioPortalProfile("keep_mutations", "keep_tcga_gdc", "MUTATION_EXTENDED", None, "Mutations"),
            mrna_profile=CbioPortalProfile("keep_mrna", "keep_tcga_gdc", "MRNA_EXPRESSION", "RNA-Seq", "mRNA"),
            mrna_raw_profile=CbioPortalProfile("keep_mrna_raw", "keep_tcga_gdc", "MRNA_EXPRESSION", "RNA-Seq", "mRNA raw"),
            mrna_transform=None,
            mutation_genes=("TP53",),
            mrna_genes=("MYC",),
        )
        mutation_table = pd.DataFrame([["Missense_Mutation", ""]], index=["TP53"], columns=["S1", "S2"])
        mrna_table = pd.DataFrame([[1.0, 0.0]], index=["MYC"], columns=["S1", "S2"])
        return CbioPortalDataBundle(source=source, mutation_table=mutation_table, mrna_table=mrna_table, mrna_raw_table=mrna_table)

    monkeypatch.setattr("geo_downloader.cbioportal_cli.load_cbioportal_study_index", lambda path: object())
    monkeypatch.setattr("geo_downloader.cbioportal_cli.resolve_all_studies_from_index", fake_resolve_all_studies_from_index)
    monkeypatch.setattr("geo_downloader.cbioportal_cli.build_cbioportal_data_source", fake_build_cbioportal_data_source)

    rc = cbioportal_main(
        [
            "--all-cancers",
            "--mutation-genes",
            "TP53",
            "--mrna-genes",
            "MYC",
            "--index-file",
            str(index_path),
            "--output-dir",
            str(tmp_path / "all"),
        ]
    )
    assert rc == 0
    assert not (tmp_path / "all" / "skip_tcga_gdc").exists()
    assert (tmp_path / "all" / "keep_tcga_gdc" / "violin.png").exists()
    assert (tmp_path / "all" / "keep_tcga_gdc" / "forest_TP53.png").exists()


def test_smoke_cbioportal_coad_script_smoke(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("scripts.smoke_cbioportal_coad.CbioPortalClient", lambda base_url: client)
    rc = smoke_cbioportal_coad_main(
        [
            "--output-dir",
            str(tmp_path / "smoke"),
            "--genes",
            "APC,TP53,KRAS,PIK3CA,SMAD4,FBXW7",
        ]
    )
    assert rc == 0
    assert (tmp_path / "smoke" / "coadread_tcga_pan_can_atlas_2018" / "oncoplot.png").exists()
    assert (tmp_path / "smoke" / "coadread_tcga_pan_can_atlas_2018" / "violin.png").exists()
    assert (tmp_path / "smoke" / "coadread_tcga_pan_can_atlas_2018" / "forest_APC.png").exists()
    assert (tmp_path / "smoke" / "coadread_tcga_pan_can_atlas_2018" / "forest_FBXW7.png").exists()
    assert (tmp_path / "smoke" / "coadread_tcga_pan_can_atlas_2018" / "association.tsv").exists()
    assert (tmp_path / "smoke" / "coadread_tcga_pan_can_atlas_2018" / "mrna_raw.tsv").exists()


def test_smoke_cbioportal_luad_script_smoke(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("scripts.smoke_cbioportal_luad.CbioPortalClient", lambda base_url: client)
    rc = smoke_cbioportal_luad_main(
        [
            "--output-dir",
            str(tmp_path / "smoke-luad"),
        ]
    )
    assert rc == 0
    output_dir = tmp_path / "smoke-luad" / "luad_tcga"
    assert (output_dir / "oncoplot.png").exists()
    assert (output_dir / "violin.png").exists()
    assert (output_dir / "forest_STK11.png").exists()
    assert (output_dir / "forest_KRAS.png").exists()
    assert (output_dir / "association.tsv").exists()
    assert (output_dir / "mrna_raw.tsv").exists()


def test_smoke_cbioportal_luad_cptac_script_smoke(tmp_path, monkeypatch):
    client = FakeCbioPortalClient()
    monkeypatch.setattr("scripts.smoke_cbioportal_luad_cptac.CbioPortalClient", lambda base_url: client)
    rc = smoke_cbioportal_luad_cptac_main(
        [
            "--output-dir",
            str(tmp_path / "smoke-luad-cptac"),
            "--mutation-profile-id",
            "toy_mutations",
            "--expression-profile-id",
            "toy_mrna",
        ]
    )
    assert rc == 0
    output_dir = tmp_path / "smoke-luad-cptac" / "luad_cptac_2020"
    assert (output_dir / "oncoplot.png").exists()
    assert (output_dir / "violin.png").exists()
    assert (output_dir / "forest_STK11.png").exists()
    assert (output_dir / "forest_KRAS.png").exists()
    assert (output_dir / "forest_TP53.png").exists()
    assert (output_dir / "association.tsv").exists()
    assert (output_dir / "mrna_raw.tsv").exists()
    assert (output_dir / "source.json").exists()


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _write_sample_archive(path: Path, sample_id: str) -> None:
    _write_multi_sample_archive(path, [sample_id])


def _write_multi_sample_archive(path: Path, sample_ids: list[str]) -> None:
    inner_bytes = io.BytesIO()
    with tarfile.open(fileobj=inner_bytes, mode="w:gz") as inner:
        matrix = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
        matrix_buf = io.BytesIO()
        mmwrite(matrix_buf, matrix)
        _add_bytes(inner, "raw_gene_bc_matrices/hg19/matrix.mtx", matrix_buf.getvalue())
        _add_text(inner, "raw_gene_bc_matrices/hg19/genes.tsv", "ENSG1\tGENE1\nENSG2\tGENE2\n")
        _add_text(inner, "raw_gene_bc_matrices/hg19/barcodes.tsv", "BC1\nBC2\n")

    with tarfile.open(path, mode="w") as outer:
        payload = inner_bytes.getvalue()
        for sample_id in sample_ids:
            info = tarfile.TarInfo(f"{sample_id}.raw_gene_bc_matrices.tar.gz")
            info.size = len(payload)
            outer.addfile(info, io.BytesIO(payload))
