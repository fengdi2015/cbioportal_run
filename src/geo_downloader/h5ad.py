from __future__ import annotations

import gzip
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix


MATRIX_BUNDLE_SUFFIXES = (
    ".raw_gene_bc_matrices.tar.gz",
    ".raw_gene_bc_matrices.tgz",
    ".raw_gene_bc_matrices.tar",
    ".gene_bc_matrices.tar.gz",
    ".gene_bc_matrices.tgz",
    ".gene_bc_matrices.tar",
    ".tar.gz",
    ".tgz",
)

FEATURE_SUFFIXES = ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz")
BARCODE_SUFFIXES = ("barcodes.tsv", "barcodes.tsv.gz")
MATRIX_SUFFIXES = ("matrix.mtx", "matrix.mtx.gz")


@dataclass(frozen=True)
class SampleBundle:
    sample_id: str
    member_name: str
    matrix_member: str
    feature_member: str
    barcode_member: str


def list_sample_bundles(outer_tar_path: Path) -> list[SampleBundle]:
    bundles: list[SampleBundle] = []
    with tarfile.open(outer_tar_path, "r:*") as outer:
        for member in outer.getmembers():
            if not member.isfile():
                continue
            if not member.name.endswith(MATRIX_BUNDLE_SUFFIXES):
                continue
            bundle = _inspect_sample_bundle(outer, member.name)
            if bundle is not None:
                bundles.append(bundle)
    return bundles


def convert_sample_bundle_to_h5ad(
    outer_tar_path: Path,
    sample_id: str,
    output_path: Path,
) -> Path:
    bundle = _find_bundle_by_sample_id(outer_tar_path, sample_id)
    adata = _build_anndata_from_bundle(outer_tar_path, bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return output_path


def convert_all_sample_bundles_to_h5ad(outer_tar_path: Path, output_dir: Path) -> list[Path]:
    bundles = list_sample_bundles(outer_tar_path)
    if not bundles:
        raise ValueError("no convertible sample bundles found in archive")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for bundle in bundles:
        out = output_dir / f"{bundle.sample_id}.h5ad"
        convert_sample_bundle_to_h5ad(outer_tar_path, bundle.sample_id, out)
        written.append(out)
    return written


def _build_anndata_from_bundle(outer_tar_path: Path, bundle: SampleBundle) -> ad.AnnData:
    with tarfile.open(outer_tar_path, "r:*") as outer:
        raw_bytes = _read_outer_member_bytes(outer, bundle.member_name)

    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:*") as inner:
        matrix_member = _find_member(inner, MATRIX_SUFFIXES)
        feature_member = _find_member(inner, FEATURE_SUFFIXES)
        barcode_member = _find_member(inner, BARCODE_SUFFIXES)

        matrix = mmread(io.BytesIO(_read_member_bytes(inner, matrix_member))).tocsr()
        features = _read_table(_read_member_bytes(inner, feature_member))
        barcodes = _read_column(_read_member_bytes(inner, barcode_member))

    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(
            f"shape mismatch for {bundle.sample_id}: matrix={matrix.shape}, features={len(features)}, barcodes={len(barcodes)}"
        )

    obs = pd.DataFrame(index=barcodes)
    obs.index.name = "barcode"
    obs["sample_id"] = bundle.sample_id
    obs["sample_bundle"] = bundle.member_name
    obs["library_id"] = bundle.sample_id

    var = _build_var_dataframe(features)

    adata = ad.AnnData(X=csr_matrix(matrix.T), obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def _inspect_sample_bundle(outer: tarfile.TarFile, member_name: str) -> SampleBundle | None:
    raw_bytes = _read_outer_member_bytes(outer, member_name)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:*") as inner:
            matrix_member = _find_member(inner, MATRIX_SUFFIXES)
            feature_member = _find_member(inner, FEATURE_SUFFIXES)
            barcode_member = _find_member(inner, BARCODE_SUFFIXES)
    except Exception:
        return None
    sample_id = _sample_id_from_member_name(member_name)
    return SampleBundle(
        sample_id=sample_id,
        member_name=member_name,
        matrix_member=matrix_member,
        feature_member=feature_member,
        barcode_member=barcode_member,
    )


def _find_bundle_by_sample_id(outer_tar_path: Path, sample_id: str) -> SampleBundle:
    bundles = list_sample_bundles(outer_tar_path)
    for bundle in bundles:
        if bundle.sample_id == sample_id:
            return bundle
    available = ", ".join(bundle.sample_id for bundle in bundles[:10])
    raise ValueError(f"sample bundle not found: {sample_id}. available examples: {available}")


def _sample_id_from_member_name(member_name: str) -> str:
    for suffix in MATRIX_BUNDLE_SUFFIXES:
        if member_name.endswith(suffix):
            return member_name[: -len(suffix)]
    return Path(member_name).stem


def _read_outer_member_bytes(outer: tarfile.TarFile, member_name: str) -> bytes:
    member = outer.getmember(member_name)
    extracted = outer.extractfile(member)
    if extracted is None:
        raise ValueError(f"could not extract bundle member: {member_name}")
    return extracted.read()


def _find_member(inner: tarfile.TarFile, suffixes: tuple[str, ...]) -> str:
    matches = [member.name for member in inner.getmembers() if member.isfile() and member.name.endswith(suffixes)]
    if not matches:
        raise ValueError(f"could not find one of {suffixes} in inner bundle")
    return matches[0]


def _read_member_bytes(inner: tarfile.TarFile, member_name: str) -> bytes:
    extracted = inner.extractfile(member_name)
    if extracted is None:
        raise ValueError(f"could not extract inner member: {member_name}")
    raw = extracted.read()
    if member_name.endswith(".gz"):
        return gzip.decompress(raw)
    return raw


def _read_table(raw: bytes) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    return rows


def _read_column(raw: bytes) -> list[str]:
    return [line.split("\t", 1)[0] for line in raw.decode("utf-8", errors="replace").splitlines() if line.strip()]


def _build_var_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    feature_ids = [row[0] if row else "" for row in rows]
    feature_names = [row[1] if len(row) > 1 and row[1] else row[0] for row in rows]
    var = pd.DataFrame(index=feature_ids)
    var.index.name = "feature_id"
    var["feature_name"] = feature_names
    var["gene_name"] = feature_names

    max_columns = max((len(row) for row in rows), default=0)
    for idx in range(2, max_columns):
        values = [row[idx] if len(row) > idx and row[idx] else None for row in rows]
        col_name = "feature_type" if idx == 2 else f"feature_col_{idx + 1}"
        if any(value is not None for value in values):
            var[col_name] = values
    return var
