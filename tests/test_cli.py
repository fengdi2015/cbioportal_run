import tarfile
import io
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from geo_downloader.cli import _write_graph
from geo_downloader.geo import build_geo_record, geo_series_download_url
from geo_downloader.h5ad import convert_all_sample_bundles_to_h5ad, convert_sample_bundle_to_h5ad, list_sample_bundles
from geo_downloader.workflow import GeoDownloadWorkflow, WorkflowSettings


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
