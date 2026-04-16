from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import mannwhitneyu


DEFAULT_BASE_URL = "https://www.cbioportal.org"
CBIOPORTAL_STUDY_INDEX_FILENAME = "cbioportal_study_index.json"


@dataclass(frozen=True)
class CbioPortalStudy:
    study_id: str
    name: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class CbioPortalProfile:
    molecular_profile_id: str
    study_id: str
    alteration_type: str | None = None
    datatype: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class CbioPortalStudyReference:
    role: str
    study: CbioPortalStudy
    sample_list_id: str
    sample_count: int | None = None
    mutation_profile_id: str | None = None
    mrna_profile_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "study": {
                "study_id": self.study.study_id,
                "name": self.study.name,
                "description": self.study.description,
            },
            "sample_list_id": self.sample_list_id,
            "sample_count": self.sample_count,
            "mutation_profile_id": self.mutation_profile_id,
            "mrna_profile_id": self.mrna_profile_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CbioPortalStudyReference":
        study_data = data.get("study") or {}
        return cls(
            role=str(data.get("role", "")),
            study=CbioPortalStudy(
                study_id=str(study_data.get("study_id") or study_data.get("studyId") or ""),
                name=study_data.get("name"),
                description=study_data.get("description"),
            ),
            sample_list_id=str(data.get("sample_list_id") or data.get("sampleListId") or ""),
            sample_count=int(data["sample_count"]) if data.get("sample_count") is not None else None,
            mutation_profile_id=data.get("mutation_profile_id"),
            mrna_profile_id=data.get("mrna_profile_id"),
        )


@dataclass(frozen=True)
class CbioPortalDataSource:
    study: CbioPortalStudy
    sample_list_id: str
    sample_ids: tuple[str, ...]
    mutation_profile: CbioPortalProfile | None
    mrna_profile: CbioPortalProfile | None
    mrna_raw_profile: CbioPortalProfile | None
    mrna_transform: str | None
    mutation_genes: tuple[str, ...]
    mrna_genes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": {
                "study_id": self.study.study_id,
                "name": self.study.name,
                "description": self.study.description,
            },
            "sample_list_id": self.sample_list_id,
            "sample_count": len(self.sample_ids),
            "sample_ids": list(self.sample_ids),
            "mutation": {
                "available": self.mutation_profile is not None and bool(self.mutation_genes),
                "profile": None if self.mutation_profile is None else {
                    "molecular_profile_id": self.mutation_profile.molecular_profile_id,
                    "alteration_type": self.mutation_profile.alteration_type,
                    "datatype": self.mutation_profile.datatype,
                    "name": self.mutation_profile.name,
                },
                "genes": list(self.mutation_genes),
            },
            "mrna": {
                "available": self.mrna_profile is not None and bool(self.mrna_genes),
                "profile": None if self.mrna_profile is None else {
                    "molecular_profile_id": self.mrna_profile.molecular_profile_id,
                    "alteration_type": self.mrna_profile.alteration_type,
                    "datatype": self.mrna_profile.datatype,
                    "name": self.mrna_profile.name,
                },
                "raw_profile": None if self.mrna_raw_profile is None else {
                    "molecular_profile_id": self.mrna_raw_profile.molecular_profile_id,
                    "alteration_type": self.mrna_raw_profile.alteration_type,
                    "datatype": self.mrna_raw_profile.datatype,
                    "name": self.mrna_raw_profile.name,
                },
                "transform": self.mrna_transform,
                "genes": list(self.mrna_genes),
            },
        }


@dataclass(frozen=True)
class CbioPortalStudySelection:
    cancer_query: str
    study_ids: tuple[str, ...]
    studies: tuple[CbioPortalStudy, ...]
    study_records: tuple[CbioPortalStudyReference, ...] = ()


@dataclass(frozen=True)
class CbioPortalStudyIndexEntry:
    cancer_id: str
    cancer_name: str
    aliases: tuple[str, ...]
    study_records: tuple[CbioPortalStudyReference, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cancer_id": self.cancer_id,
            "cancer_name": self.cancer_name,
            "aliases": list(self.aliases),
            "study_records": [record.to_dict() for record in self.study_records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CbioPortalStudyIndexEntry":
        return cls(
            cancer_id=str(data.get("cancer_id", "")),
            cancer_name=str(data.get("cancer_name", "")),
            aliases=tuple(str(alias) for alias in data.get("aliases", []) if str(alias).strip()),
            study_records=tuple(
                CbioPortalStudyReference.from_dict(item)
                for item in data.get("study_records", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class CbioPortalStudyIndex:
    schema_version: int
    base_url: str
    generated_at: str
    entries: tuple[CbioPortalStudyIndexEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_url": self.base_url,
            "generated_at": self.generated_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CbioPortalStudyIndex":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            base_url=str(data.get("base_url", DEFAULT_BASE_URL)),
            generated_at=str(data.get("generated_at", "")),
            entries=tuple(
                CbioPortalStudyIndexEntry.from_dict(item)
                for item in data.get("entries", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class CbioPortalPlotResult:
    study: CbioPortalStudy
    sample_list_id: str
    sample_ids: tuple[str, ...]
    mutation_table: pd.DataFrame
    mrna_table: pd.DataFrame
    output_plot: Path
    output_dir: Path
    mutation_table_path: Path
    mrna_table_path: Path
    mrna_raw_table_path: Path | None
    samples_path: Path
    association_table_path: Path | None = None
    source_path: Path | None = None


class CbioPortalClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0"})

    def get_json(self, path: str, *, params: dict[str, Any] | None = None, method: str = "GET", json_body: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def get_study(self, study_id: str) -> CbioPortalStudy:
        data = self.get_json(f"/api/studies/{study_id}")
        return CbioPortalStudy(
            study_id=_pick(data, "studyId", "study_id", default=study_id),
            name=_pick(data, "name"),
            description=_pick(data, "description"),
        )

    def get_sample_lists(self, study_id: str) -> list[dict[str, Any]]:
        data = self.get_json(f"/api/studies/{study_id}/sample-lists", params={"projection": "SUMMARY", "pageSize": 10000000})
        return list(data or [])

    def get_studies(self) -> list[dict[str, Any]]:
        data = self.get_json("/api/studies", params={"projection": "SUMMARY", "pageSize": 10000000})
        return list(data or [])

    def get_sample_list(self, sample_list_id: str) -> list[str]:
        data = self.get_json(f"/api/sample-lists/{sample_list_id}")
        sample_ids = _pick(data, "sampleIds", "sample_ids", default=[])
        return [str(sample_id) for sample_id in sample_ids]

    def get_molecular_profiles(self, study_id: str) -> list[CbioPortalProfile]:
        data = self.get_json("/api/molecular-profiles", params={"projection": "SUMMARY", "pageSize": 10000000})
        profiles = []
        for item in data or []:
            if _pick(item, "studyId", "study_id") != study_id:
                continue
            profiles.append(
                CbioPortalProfile(
                    molecular_profile_id=_pick(item, "molecularProfileId", "molecular_profile_id"),
                    study_id=study_id,
                    alteration_type=_pick(item, "molecularAlterationType", "alterationType"),
                    datatype=_pick(item, "datatype"),
                    name=_pick(item, "name"),
                )
            )
        return sorted(profiles, key=lambda profile: profile.molecular_profile_id.lower())

    def resolve_sample_list_id(self, study_id: str, sample_list_id: str | None = None) -> str:
        sample_lists = self.get_sample_lists(study_id)
        if sample_list_id:
            return sample_list_id
        preferred = f"{study_id}_all"
        for item in sample_lists:
            if _pick(item, "sampleListId", "sample_list_id") == preferred:
                return preferred
        for item in sample_lists:
            category = str(_pick(item, "category", default="")).lower()
            if category == "all_cases_in_study":
                return str(_pick(item, "sampleListId", "sample_list_id"))
        if sample_lists:
            return str(_pick(sample_lists[0], "sampleListId", "sample_list_id"))
        raise ValueError(f"no sample lists found for study {study_id}")

    def resolve_sample_ids(self, study_id: str, sample_list_id: str | None = None) -> tuple[str, str]:
        resolved = self.resolve_sample_list_id(study_id, sample_list_id)
        sample_ids = self.get_sample_list(resolved)
        if not sample_ids:
            raise ValueError(f"sample list {resolved} does not contain any sample IDs")
        return resolved, tuple(sample_ids)

    def resolve_profile(
        self,
        study_id: str,
        kind: str,
        exact_profile_id: str | None = None,
        prefer_zscore: bool = True,
    ) -> CbioPortalProfile:
        profile = self.find_profile(study_id, kind, exact_profile_id=exact_profile_id, prefer_zscore=prefer_zscore)
        if profile is None:
            raise ValueError(f"could not resolve {kind} profile for study {study_id}")
        return profile

    def find_profile(
        self,
        study_id: str,
        kind: str,
        exact_profile_id: str | None = None,
        prefer_zscore: bool = True,
    ) -> CbioPortalProfile | None:
        profiles = self.get_molecular_profiles(study_id)
        if exact_profile_id:
            exact_profile_id_lower = exact_profile_id.lower()
            for profile in profiles:
                if profile.molecular_profile_id.lower() == exact_profile_id_lower:
                    return profile
            return None
        kind_lower = kind.lower()
        candidates: list[tuple[tuple[int, int, int, int, str], CbioPortalProfile]] = []
        for profile in profiles:
            alteration = str(profile.alteration_type or "").upper()
            datatype = str(profile.datatype or "").upper()
            profile_id = profile.molecular_profile_id.lower()
            if kind_lower == "mutation":
                if alteration == "MUTATION_EXTENDED" or profile_id.endswith("_mutations"):
                    priority = (0, 0, 0, 0, profile_id)
                    candidates.append((priority, profile))
                continue
            if kind_lower in {"mrna", "expression"}:
                if alteration != "MRNA_EXPRESSION" or not (profile_id.endswith("_mrna") or "_rna_seq_" in profile_id):
                    continue
                is_zscore = "z-score" in datatype.lower() or "zscore" in profile_id
                if prefer_zscore:
                    priority = (
                        0 if is_zscore else 1,
                        0 if "_rna_seq_" in profile_id else 1,
                        0 if profile_id.endswith("_mrna") else 1,
                        0 if "median" in profile_id else 1,
                        profile_id,
                    )
                else:
                    priority = (
                        0 if not is_zscore else 1,
                        0 if "_rna_seq_" in profile_id else 1,
                        0 if profile_id.endswith("_mrna") else 1,
                        0 if "median" in profile_id else 1,
                        profile_id,
                    )
                candidates.append((priority, profile))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def resolve_default_studies_for_cancer(
        self,
        cancer_query: str,
        study_index: CbioPortalStudyIndex | None = None,
    ) -> CbioPortalStudySelection:
        query = cancer_query.strip().lower()
        if not query:
            raise ValueError("cancer query cannot be empty")

        index_was_available = study_index is not None
        if study_index is None:
            default_index_path = default_cbioportal_study_index_path()
            if default_index_path.exists():
                study_index = load_cbioportal_study_index(default_index_path)
                index_was_available = True
        if study_index is not None:
            selection = resolve_studies_for_cancer_from_index(study_index, cancer_query)
            if selection is not None:
                return selection
            if index_was_available:
                raise ValueError(f"no studies matched cancer query {cancer_query!r} in the frozen cBioPortal index")

        studies = self.get_studies()
        matches = [
            study
            for study in studies
            if query in str(_pick(study, "studyId", "study_id", default="")).lower()
            or query in str(_pick(study, "name", default="")).lower()
        ]
        if not matches:
            raise ValueError(f"no studies matched cancer query {cancer_query!r}")

        tcga_gdc = _select_tcga_gdc_2025(matches)
        if tcga_gdc is None:
            raise ValueError(f"could not find a TCGA GDC 2025 study for {cancer_query!r}")

        resolved_studies = [self.get_study(str(_pick(tcga_gdc, "studyId", "study_id")))]
        cptac = _select_cptac_study(matches)
        if cptac is not None:
            cptac_id = str(_pick(cptac, "studyId", "study_id"))
            if cptac_id != resolved_studies[0].study_id:
                resolved_studies.append(self.get_study(cptac_id))

        return CbioPortalStudySelection(
            cancer_query=cancer_query,
            study_ids=tuple(study.study_id for study in resolved_studies),
            studies=tuple(resolved_studies),
            study_records=tuple(),
        )

    def resolve_gene(self, gene: str) -> dict[str, Any]:
        data = self.get_json(f"/api/genes/{gene}")
        if not isinstance(data, dict):
            raise ValueError(f"gene lookup failed for {gene}")
        return data

    def fetch_mutations(self, molecular_profile_id: str, sample_list_id: str, entrez_gene_id: int) -> pd.DataFrame:
        rows = self.get_json(
            f"/api/molecular-profiles/{molecular_profile_id}/mutations",
            params={
                "sampleListId": sample_list_id,
                "entrezGeneId": entrez_gene_id,
                "projection": "DETAILED",
                "pageSize": 10000000,
            },
        )
        return pd.DataFrame(rows or [])

    def fetch_expression(self, molecular_profile_id: str, sample_list_id: str, entrez_gene_id: int) -> pd.DataFrame:
        rows = self.get_json(
            f"/api/molecular-profiles/{molecular_profile_id}/molecular-data",
            params={
                "sampleListId": sample_list_id,
                "entrezGeneId": entrez_gene_id,
                "projection": "SUMMARY",
            },
        )
        return pd.DataFrame(rows or [])


def default_cbioportal_study_index_path() -> Path:
    return Path(__file__).with_name(CBIOPORTAL_STUDY_INDEX_FILENAME)


def load_cbioportal_study_index(path: Path | None = None) -> CbioPortalStudyIndex:
    resolved_path = path or default_cbioportal_study_index_path()
    with resolved_path.open("r", encoding="utf-8") as handle:
        return CbioPortalStudyIndex.from_dict(json.load(handle))


def save_cbioportal_study_index(index: CbioPortalStudyIndex, path: Path | None = None) -> Path:
    resolved_path = path or default_cbioportal_study_index_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(index.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved_path


def build_cbioportal_study_index(client: CbioPortalClient) -> CbioPortalStudyIndex:
    studies = client.get_studies()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for study in studies:
        study_id = str(_pick(study, "studyId", "study_id", default="")).strip()
        if not study_id:
            continue
        cancer_id = _infer_cancer_id_from_study_id(study_id)
        grouped.setdefault(cancer_id, []).append(study)

    entries: list[CbioPortalStudyIndexEntry] = []
    for cancer_id, group in sorted(grouped.items()):
        tcga = _select_tcga_gdc_2025(group)
        if tcga is None:
            continue
        tcga_id = str(_pick(tcga, "studyId", "study_id"))
        tcga_study = client.get_study(tcga_id)
        tcga_reference = _build_study_reference(client, tcga_study, role="tcga_gdc_2025")
        study_records = [tcga_reference]

        cptac = _select_cptac_study(group)
        if cptac is not None:
            cptac_id = str(_pick(cptac, "studyId", "study_id"))
            if cptac_id != tcga_id:
                cptac_study = client.get_study(cptac_id)
                study_records.append(_build_study_reference(client, cptac_study, role="cptac"))

        cancer_name = _infer_cancer_name(tcga_study.name or str(_pick(tcga, "name", default="")) or cancer_id)
        aliases = tuple(
            sorted(
                {
                    cancer_id,
                    _normalize_cancer_token(cancer_id),
                    _normalize_cancer_token(cancer_name),
                    _normalize_cancer_token(tcga_study.study_id),
                }
            )
        )
        entries.append(
            CbioPortalStudyIndexEntry(
                cancer_id=cancer_id,
                cancer_name=cancer_name,
                aliases=aliases,
                study_records=tuple(study_records),
            )
        )

    return CbioPortalStudyIndex(
        schema_version=1,
        base_url=getattr(client, "base_url", DEFAULT_BASE_URL),
        generated_at=datetime.now(timezone.utc).isoformat(),
        entries=tuple(entries),
    )


def resolve_studies_for_cancer_from_index(
    study_index: CbioPortalStudyIndex,
    cancer_query: str,
) -> CbioPortalStudySelection | None:
    query = _normalize_cancer_token(cancer_query)
    if not query:
        raise ValueError("cancer query cannot be empty")
    for entry in study_index.entries:
        candidates = {
            _normalize_cancer_token(entry.cancer_id),
            _normalize_cancer_token(entry.cancer_name),
            *(_normalize_cancer_token(alias) for alias in entry.aliases),
        }
        if not any(query in candidate or candidate in query for candidate in candidates):
            continue
        studies = tuple(record.study for record in entry.study_records)
        return CbioPortalStudySelection(
            cancer_query=cancer_query,
            study_ids=tuple(study.study_id for study in studies),
            studies=studies,
            study_records=entry.study_records,
        )
    return None


def _build_study_reference(client: CbioPortalClient, study: CbioPortalStudy, role: str) -> CbioPortalStudyReference:
    sample_list_id = client.resolve_sample_list_id(study.study_id)
    sample_ids = client.get_sample_list(sample_list_id)
    mutation_profile = client.find_profile(study.study_id, "mutation")
    mrna_profile = client.find_profile(study.study_id, "mrna", prefer_zscore=not _is_tcga_study(study))
    return CbioPortalStudyReference(
        role=role,
        study=study,
        sample_list_id=sample_list_id,
        sample_count=len(sample_ids),
        mutation_profile_id=None if mutation_profile is None else mutation_profile.molecular_profile_id,
        mrna_profile_id=None if mrna_profile is None else mrna_profile.molecular_profile_id,
    )


def _infer_cancer_id_from_study_id(study_id: str) -> str:
    study_id = study_id.lower()
    for suffix in ("_tcga_gdc", "_cptac_2025", "_cptac_2021", "_cptac_2020", "_cptac_2019", "_cptac_2018", "_cptac"):
        if suffix in study_id:
            return study_id.split(suffix, 1)[0]
    if "_tcga_" in study_id:
        return study_id.split("_tcga_", 1)[0]
    if "_" in study_id:
        return study_id.split("_", 1)[0]
    return study_id


def _is_tcga_study(study: CbioPortalStudy) -> bool:
    study_id = study.study_id.lower()
    study_name = (study.name or "").lower()
    return "_tcga_" in study_id or "tcga gdc" in study_name


def _infer_cancer_name(text: str) -> str:
    cleaned = str(text).strip()
    if "(" in cleaned:
        cleaned = cleaned.split("(", 1)[0].strip()
    return cleaned


def _normalize_cancer_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def build_cbioportal_matrices(
    client: CbioPortalClient,
    study_id: str,
    mutation_genes: Iterable[str],
    mrna_genes: Iterable[str] | None = None,
    sample_list_id: str | None = None,
) -> tuple[CbioPortalStudy, str, list[str], pd.DataFrame, pd.DataFrame]:
    bundle = build_cbioportal_data_source(
        client=client,
        study_id=study_id,
        mutation_genes=mutation_genes,
        mrna_genes=mrna_genes,
        sample_list_id=sample_list_id,
    )
    mutation_table = bundle.mutation_table
    mrna_table = bundle.mrna_table
    if mutation_table is None or mrna_table is None:
        raise ValueError("both mutation and mRNA tables are required for build_cbioportal_matrices")
    return bundle.source.study, bundle.source.sample_list_id, list(bundle.source.sample_ids), mutation_table, mrna_table


@dataclass(frozen=True)
class CbioPortalDataBundle:
    source: CbioPortalDataSource
    mutation_table: pd.DataFrame | None
    mrna_table: pd.DataFrame | None
    mrna_raw_table: pd.DataFrame | None


def build_cbioportal_data_source(
    client: CbioPortalClient,
    study_id: str,
    mutation_genes: Iterable[str] | None = None,
    mrna_genes: Iterable[str] | None = None,
    sample_list_id: str | None = None,
    mutation_profile_id: str | None = None,
    mrna_profile_id: str | None = None,
    mrna_raw_profile_id: str | None = None,
    study: CbioPortalStudy | None = None,
) -> CbioPortalDataBundle:
    study = study or client.get_study(study_id)
    resolved_sample_list_id, sample_ids = client.resolve_sample_ids(study_id, sample_list_id)
    is_tcga = _is_tcga_study(study)
    mutation_profile = client.find_profile(study_id, "mutation", exact_profile_id=mutation_profile_id)
    mrna_profile = client.find_profile(study_id, "mrna", exact_profile_id=mrna_profile_id, prefer_zscore=True)
    mrna_raw_profile = client.find_profile(study_id, "mrna", exact_profile_id=mrna_raw_profile_id, prefer_zscore=False)
    mutation_genes = tuple(str(gene).strip() for gene in (mutation_genes or []) if str(gene).strip())
    mrna_genes = tuple(str(gene).strip() for gene in (mrna_genes or []) if str(gene).strip())

    mutation_table = None
    if mutation_genes:
        if mutation_profile is None:
            raise ValueError(f"study {study_id} does not provide a mutation profile")
        mutation_table = _build_modality_table(
            client=client,
            profile=mutation_profile,
            sample_list_id=resolved_sample_list_id,
            sample_ids=sample_ids,
            genes=mutation_genes,
            modality="mutation",
        )

    mrna_table = None
    mrna_raw_table = None
    if mrna_genes:
        if mrna_raw_profile is None:
            raise ValueError(f"study {study_id} does not provide an mRNA profile")
        mrna_raw_table = _build_modality_table(
            client=client,
            profile=mrna_raw_profile,
            sample_list_id=resolved_sample_list_id,
            sample_ids=sample_ids,
            genes=mrna_genes,
            modality="mrna",
        )
        if mrna_profile is None:
            mrna_profile = mrna_raw_profile
        mrna_table = _build_modality_table(
            client=client,
            profile=mrna_profile,
            sample_list_id=resolved_sample_list_id,
            sample_ids=sample_ids,
            genes=mrna_genes,
            modality="mrna",
        )
        if is_tcga or str(mrna_profile.datatype or "").upper() != "Z-SCORE":
            mrna_table = _zscore_rows(mrna_raw_table)

    source = CbioPortalDataSource(
        study=study,
        sample_list_id=resolved_sample_list_id,
        sample_ids=tuple(sample_ids),
        mutation_profile=mutation_profile,
        mrna_profile=mrna_profile,
        mrna_raw_profile=mrna_raw_profile,
        mrna_transform="zscore" if is_tcga and mrna_genes else None,
        mutation_genes=mutation_genes,
        mrna_genes=mrna_genes,
    )
    return CbioPortalDataBundle(source=source, mutation_table=mutation_table, mrna_table=mrna_table, mrna_raw_table=mrna_raw_table)


def _build_modality_table(
    client: CbioPortalClient,
    profile: CbioPortalProfile,
    sample_list_id: str,
    sample_ids: tuple[str, ...],
    genes: tuple[str, ...],
    modality: str,
    expression_transform: str | None = None,
) -> pd.DataFrame:
    table = pd.DataFrame(index=list(genes), columns=sample_ids, dtype=object if modality == "mutation" else float)
    for gene in genes:
        gene_info = client.resolve_gene(gene)
        entrez_gene_id = _pick(gene_info, "entrezGeneId", "geneId", "entrez_gene_id")
        if entrez_gene_id is None:
            raise ValueError(f"gene {gene} did not resolve to an Entrez gene id")
        if modality == "mutation":
            rows = client.fetch_mutations(profile.molecular_profile_id, sample_list_id, int(entrez_gene_id))
            if rows.empty:
                table.loc[gene] = ""
                continue
            sample_col = _sample_column(rows)
            type_col = _mutation_type_column(rows)
            for sample_id in sample_ids:
                hits = rows[rows[sample_col].astype(str) == sample_id]
                table.loc[gene, sample_id] = _summarize_mutation_types(hits, type_col)
            continue
        rows = client.fetch_expression(profile.molecular_profile_id, sample_list_id, int(entrez_gene_id))
        if rows.empty:
            table.loc[gene] = np.nan
            continue
        sample_col = _sample_column(rows)
        value_col = _value_column(rows)
        values = rows[[sample_col, value_col]].copy()
        values[sample_col] = values[sample_col].astype(str)
        for sample_id in sample_ids:
            match = values[values[sample_col] == sample_id]
            table.loc[gene, sample_id] = float(match.iloc[0][value_col]) if not match.empty else np.nan
    if modality == "mrna" and expression_transform == "zscore":
        table = _zscore_rows(table)
    return table


def _zscore_rows(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.apply(pd.to_numeric, errors="coerce").astype(float)
    for index, row in normalized.iterrows():
        values = row.to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            normalized.loc[index] = np.nan
            continue
        finite_values = values[finite]
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values, ddof=0))
        if std == 0:
            normalized.loc[index, finite] = 0.0
            normalized.loc[index, ~finite] = np.nan
            continue
        normalized.loc[index, finite] = (finite_values - mean) / std
        normalized.loc[index, ~finite] = np.nan
    return normalized


def save_cbioportal_outputs(
    output_dir: Path,
    study: CbioPortalStudy,
    sample_list_id: str,
    sample_ids: list[str],
    mutation_table: pd.DataFrame,
    mrna_table: pd.DataFrame,
    output_plot: Path,
    mrna_raw_table: pd.DataFrame | None = None,
    association_table: pd.DataFrame | None = None,
    source: CbioPortalDataSource | None = None,
) -> CbioPortalPlotResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    mutation_table_path = output_dir / "mutations.tsv"
    mrna_table_path = output_dir / "mrna.tsv"
    mrna_raw_table_path = output_dir / "mrna_raw.tsv" if mrna_raw_table is not None else None
    samples_path = output_dir / "samples.txt"
    association_table_path = output_dir / "association.tsv" if association_table is not None else None
    source_path = output_dir / "source.json" if source is not None else None
    mutation_table.to_csv(mutation_table_path, sep="\t")
    mrna_table.to_csv(mrna_table_path, sep="\t")
    if mrna_raw_table_path is not None and mrna_raw_table is not None:
        mrna_raw_table.to_csv(mrna_raw_table_path, sep="\t")
    samples_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    if association_table_path is not None:
        association_table.to_csv(association_table_path, sep="\t", index=False)
    if source_path is not None and source is not None:
        source_path.write_text(json.dumps(source.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return CbioPortalPlotResult(
        study=study,
        sample_list_id=sample_list_id,
        sample_ids=tuple(sample_ids),
        mutation_table=mutation_table,
        mrna_table=mrna_table,
        output_plot=output_plot,
        output_dir=output_dir,
        mutation_table_path=mutation_table_path,
        mrna_table_path=mrna_table_path,
        mrna_raw_table_path=mrna_raw_table_path,
        samples_path=samples_path,
        association_table_path=association_table_path,
        source_path=source_path,
    )


def plot_cbioportal_oncoplot(
    output_path: Path,
    mutation_table: pd.DataFrame | None = None,
    mrna_table: pd.DataFrame | None = None,
    source: CbioPortalDataSource | None = None,
    title: str | None = None,
    expression_label: str = "mRNA score",
    presentation: bool = False,
    max_sample_labels: int = 20,
    hide_sample_labels: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mutation_table is None and mrna_table is None:
        raise ValueError("at least one of mutation_table or mrna_table must be provided")

    reference_table = mutation_table if mutation_table is not None else mrna_table
    sample_labels = list(reference_table.columns) if reference_table is not None else []
    mutation_panel = mutation_table is not None and not mutation_table.empty
    mrna_panel = mrna_table is not None and not mrna_table.empty

    if presentation:
        sample_order = _order_samples_for_presentation(mutation_table, mrna_table)
        if mutation_table is not None:
            mutation_table = mutation_table.loc[:, sample_order]
        if mrna_table is not None:
            mrna_table = mrna_table.loc[:, sample_order]
        reference_table = mutation_table if mutation_table is not None else mrna_table
        sample_labels = list(reference_table.columns) if reference_table is not None else []
        if hide_sample_labels:
            tick_positions, tick_labels = [], []
        else:
            tick_positions, tick_labels = _sparse_sample_ticks(sample_labels, max_sample_labels=max_sample_labels)
    else:
        tick_positions = list(range(len(sample_labels)))
        tick_labels = sample_labels

    panels: list[tuple[str, np.ndarray, list[str], Callable[..., None]]] = []
    if mutation_panel and mutation_table is not None:
        mutation_rows = list(mutation_table.index)
        mutation_matrix = np.array([[1 if _is_mutated(value) else 0 for value in mutation_table.loc[row]] for row in mutation_rows], dtype=float)
        panels.append(("Mutation", mutation_matrix, mutation_rows, _draw_binary_heatmap))
    if mrna_panel and mrna_table is not None:
        mrna_rows = list(mrna_table.index)
        mrna_matrix = mrna_table.fillna(np.nan).to_numpy(dtype=float)
        panels.append((expression_label, mrna_matrix, mrna_rows, _draw_continuous_heatmap))

    if not panels:
        raise ValueError("no data available to plot")

    fig_height = max(3.0, 0.32 * sum(len(rows) for _, _, rows, _ in panels) + 1.8)
    fig_width = 8.4 if presentation else max(8.0, 0.35 * max(len(sample_labels), 8))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [max(1, len(rows)) for _, _, rows, _ in panels]},
    )

    for axis, (label, matrix, rows, drawer) in zip(axes[:, 0], panels):
        drawer(axis, matrix, rows, sample_labels, label, tick_positions, tick_labels)
    if title:
        fig.suptitle(title, fontsize=14, y=0.99)
    footer = _format_source_footer(source)
    if footer:
        fig.text(0.01, 0.01, footer, fontsize=8, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=220 if presentation else 200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_mutation_expression_associations(
    mutation_table: pd.DataFrame,
    mrna_table: pd.DataFrame,
    mrna_raw_table: pd.DataFrame | None = None,
    mutation_genes: Iterable[str] | None = None,
    expression_genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mutation_genes = list(mutation_genes or mutation_table.index)
    expression_genes = list(expression_genes or mrna_table.index)
    source_table = mrna_raw_table if mrna_raw_table is not None else mrna_table
    for mutation_gene in mutation_genes:
        if mutation_gene not in mutation_table.index:
            continue
        mutation_mask = mutation_table.loc[mutation_gene].map(_is_mutated).astype(bool)
        for expression_gene in expression_genes:
            if expression_gene not in source_table.index:
                continue
            expression = pd.to_numeric(source_table.loc[expression_gene], errors="coerce")
            mutated = expression[mutation_mask].dropna()
            wild_type = expression[~mutation_mask].dropna()
            if mutated.empty or wild_type.empty:
                continue
            stat = mannwhitneyu(mutated, wild_type, alternative="two-sided")
            rows.append(
                {
                    "mutation_gene": mutation_gene,
                    "expression_gene": expression_gene,
                    "mutated_n": int(mutated.size),
                    "wild_type_n": int(wild_type.size),
                    "mutated_median": float(mutated.median()),
                    "wild_type_median": float(wild_type.median()),
                    "delta_median": float(mutated.median() - wild_type.median()),
                    "p_value": float(stat.pvalue),
                    "neg_log10_p": float(-np.log10(stat.pvalue)) if stat.pvalue > 0 else np.inf,
                }
            )
    return pd.DataFrame(rows)


def plot_mutation_expression_relationships(
    output_path: Path,
    mutation_table: pd.DataFrame,
    mrna_table: pd.DataFrame,
    mrna_raw_table: pd.DataFrame | None = None,
    source: CbioPortalDataSource | None = None,
    title: str | None = None,
    association_table: pd.DataFrame | None = None,
    summary_label: str = "Mutation-expression summary",
    presentation: bool = True,
    n_bootstrap: int = 400,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_table = mrna_raw_table if mrna_raw_table is not None else mrna_table
    if mutation_table.empty or source_table.empty:
        raise ValueError("both mutation_table and mrna_table must contain data for relationship plotting")

    mutation_genes = list(mutation_table.index)
    expression_genes = list(source_table.index)
    association_table = association_table if association_table is not None else compute_mutation_expression_associations(
        mutation_table,
        mrna_table,
        mrna_raw_table=source_table,
    )
    if association_table.empty:
        raise ValueError("no mutation-expression associations available to plot")

    n_rows = len(mutation_genes) + 1
    n_cols = max(1, len(expression_genes))
    fig_width = 3.8 * n_cols if presentation else max(10.0, 3.2 * n_cols)
    fig_height = 2.7 * len(mutation_genes) + max(2.8, 0.42 * len(association_table) + 1.2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        height_ratios=[1.0] * len(mutation_genes) + [max(1.6, 0.4 * len(association_table))],
        hspace=0.6,
        wspace=0.35,
    )

    mutation_palette = {
        gene: color
        for gene, color in zip(mutation_genes, plt.cm.Set2(np.linspace(0, 1, max(len(mutation_genes), 3))))
    }
    rng = np.random.default_rng(42)
    for row_index, mutation_gene in enumerate(mutation_genes):
        mutation_mask = mutation_table.loc[mutation_gene].map(_is_mutated).astype(bool)
        for col_index, expression_gene in enumerate(expression_genes):
            ax = fig.add_subplot(gs[row_index, col_index])
            values = pd.to_numeric(source_table.loc[expression_gene], errors="coerce")
            wt = values[~mutation_mask].dropna().to_numpy(dtype=float)
            mut = values[mutation_mask].dropna().to_numpy(dtype=float)
            _draw_violin_jitter_panel(
                ax,
                wild_type=wt,
                mutant=mut,
                mutation_gene=mutation_gene,
                expression_gene=expression_gene,
                mutation_color=mutation_palette.get(mutation_gene, "#e45756"),
            )
            if row_index == len(mutation_genes) - 1:
                ax.set_xlabel("Mutation status")
            else:
                ax.set_xlabel("")
            if col_index == 0:
                ax.set_ylabel("mRNA log2(TPM)")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if len(expression_genes) > 1:
                ax.set_title(expression_gene, fontsize=10)
            if len(wt) and len(mut):
                jitter_wt = rng.normal(0.0, 0.05, size=len(wt))
                jitter_mut = rng.normal(1.0, 0.05, size=len(mut))
                ax.scatter(jitter_wt, wt, s=12, alpha=0.6, color="#4c78a8", edgecolors="none")
                ax.scatter(jitter_mut, mut, s=12, alpha=0.6, color=mutation_palette.get(mutation_gene, "#e45756"), edgecolors="none")
            ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"WT (n={len(wt)})", f"Mut (n={len(mut)})"], fontsize=8, rotation=0)

    summary_ax = fig.add_subplot(gs[len(mutation_genes), :])
    _draw_forest_summary_panel(
        summary_ax,
        association_table=association_table,
        mutation_table=mutation_table,
        mrna_table=source_table,
        mutation_palette=mutation_palette,
        n_bootstrap=n_bootstrap,
        summary_label=summary_label,
    )

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    footer = _format_source_footer(source)
    if footer:
        fig.text(0.01, 0.01, footer, fontsize=8, ha="left", va="bottom")
    else:
        pass
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.08, hspace=0.7, wspace=0.35)
    fig.savefig(output_path, dpi=220 if presentation else 200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_violin_jitter_panel(
    ax: plt.Axes,
    wild_type: np.ndarray,
    mutant: np.ndarray,
    mutation_gene: str,
    expression_gene: str,
    mutation_color: str,
) -> None:
    ax.set_title(f"{mutation_gene} vs {expression_gene}", fontsize=9)
    if wild_type.size == 0 and mutant.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["WT", "Mut"])
        return

    positions = [0, 1]
    groups = [wild_type, mutant]
    colors = ["#4c78a8", mutation_color]
    if wild_type.size and mutant.size:
        parts = ax.violinplot(groups, positions=positions, widths=0.8, showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.28)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#333333")
            parts["cmedians"].set_linewidth(1.2)
    else:
        data = wild_type if wild_type.size else mutant
        pos = positions[0] if wild_type.size else positions[1]
        ax.boxplot(
            [data],
            positions=[pos],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colors[0] if wild_type.size else colors[1], "alpha": 0.25, "edgecolor": "#333333"},
            medianprops={"color": "#333333", "linewidth": 1.2},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["WT", "Mut"])
    ax.set_xlim(-0.6, 1.6)
    if wild_type.size or mutant.size:
        combined = np.concatenate([wild_type, mutant]) if wild_type.size and mutant.size else (wild_type if wild_type.size else mutant)
        finite = combined[np.isfinite(combined)]
        if finite.size:
            pad = max(0.5, 0.12 * float(np.nanmax(finite) - np.nanmin(finite)))
            ax.set_ylim(float(np.nanmin(finite)) - pad, float(np.nanmax(finite)) + pad)


def _draw_forest_summary_panel(
    ax: plt.Axes,
    association_table: pd.DataFrame,
    mutation_table: pd.DataFrame,
    mrna_table: pd.DataFrame,
    mutation_palette: dict[str, str],
    n_bootstrap: int,
    summary_label: str,
) -> None:
    rows: list[dict[str, Any]] = []
    ordered = association_table.sort_values(["delta_median", "neg_log10_p"], ascending=[False, False]).copy()
    for _, row in ordered.iterrows():
        mutation_gene = str(row["mutation_gene"])
        expression_gene = str(row["expression_gene"])
        mutation_mask = mutation_table.loc[mutation_gene].map(_is_mutated).astype(bool)
        expression = pd.to_numeric(mrna_table.loc[expression_gene], errors="coerce")
        mutated = expression[mutation_mask].dropna().to_numpy(dtype=float)
        wild_type = expression[~mutation_mask].dropna().to_numpy(dtype=float)
        if mutated.size == 0 or wild_type.size == 0:
            continue
        ci_low, ci_high = _bootstrap_delta_median_ci(mutated, wild_type, n_bootstrap=n_bootstrap, seed=abs(hash((mutation_gene, expression_gene))) % (2**32))
        rows.append(
            {
                "label": f"{mutation_gene} vs {expression_gene}",
                "mutation_gene": mutation_gene,
                "expression_gene": expression_gene,
                "delta_median": float(row["delta_median"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(row["p_value"]),
            }
        )

    if not rows:
        ax.text(0.5, 0.5, "no summary data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    plot_rows = sorted(rows, key=lambda item: (item["mutation_gene"], item["delta_median"], item["expression_gene"]))
    y_positions = np.arange(len(plot_rows))[::-1]
    for y, row in zip(y_positions, plot_rows):
        color = mutation_palette.get(row["mutation_gene"], "#e45756")
        ax.errorbar(
            row["delta_median"],
            y,
            xerr=[[row["delta_median"] - row["ci_low"]], [row["ci_high"] - row["delta_median"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
        )
        ax.text(
            0.01,
            y,
            f"{row['label']}  p={row['p_value']:.2g}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel("Delta median mRNA Z-score (mutant - wild-type)")
    ax.set_title(summary_label, fontsize=10)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_ylim(-1, len(plot_rows))


def _bootstrap_delta_median_ci(
    mutated: np.ndarray,
    wild_type: np.ndarray,
    n_bootstrap: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    if mutated.size == 0 or wild_type.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        mutated_sample = rng.choice(mutated, size=mutated.size, replace=True)
        wild_type_sample = rng.choice(wild_type, size=wild_type.size, replace=True)
        deltas[index] = float(np.median(mutated_sample) - np.median(wild_type_sample))
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def _draw_binary_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    rows: list[str],
    cols: list[str],
    label: str,
    tick_positions: list[int] | None = None,
    tick_labels: list[str] | None = None,
) -> None:
    cmap = ListedColormap(["#f7f7f7", "#b2182b"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(tick_positions if tick_positions is not None else range(len(cols)))
    ax.set_xticklabels(tick_labels if tick_labels is not None else cols, rotation=90, fontsize=8)
    ax.set_ylabel(label)
    ax.set_xlim(-0.5, max(len(cols) - 0.5, 0.5))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", length=0)


def _draw_continuous_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    rows: list[str],
    cols: list[str],
    label: str,
    tick_positions: list[int] | None = None,
    tick_labels: list[str] | None = None,
) -> None:
    cmap = LinearSegmentedColormap.from_list("mrna", ["#2166ac", "#f7f7f7", "#b2182b"])
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        max_abs = float(np.nanmax(np.abs(finite)))
        if max_abs == 0:
            max_abs = 1.0
    else:
        max_abs = 1.0
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(tick_positions if tick_positions is not None else range(len(cols)))
    ax.set_xticklabels(tick_labels if tick_labels is not None else cols, rotation=90, fontsize=8)
    ax.set_ylabel(label)
    ax.set_xlim(-0.5, max(len(cols) - 0.5, 0.5))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", length=0)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)


def _sparse_sample_ticks(sample_labels: list[str], max_sample_labels: int = 20) -> tuple[list[int], list[str]]:
    if len(sample_labels) <= max_sample_labels:
        return list(range(len(sample_labels))), sample_labels
    step = int(np.ceil(len(sample_labels) / max_sample_labels))
    positions = list(range(0, len(sample_labels), step))
    labels = [sample_labels[index] for index in positions]
    return positions, labels


def _order_samples_for_presentation(
    mutation_table: pd.DataFrame | None,
    mrna_table: pd.DataFrame | None,
) -> list[str]:
    if mutation_table is None and mrna_table is None:
        return []
    reference_table = mutation_table if mutation_table is not None else mrna_table
    assert reference_table is not None
    scores = pd.DataFrame(index=reference_table.columns)
    if mutation_table is not None:
        scores["mutation_burden"] = mutation_table.apply(lambda column: column.map(_is_mutated)).sum(axis=0)
    else:
        scores["mutation_burden"] = 0
    if mrna_table is not None:
        scores["mrna_signal"] = mrna_table.apply(pd.to_numeric, errors="coerce").fillna(0).abs().sum(axis=0)
    else:
        scores["mrna_signal"] = 0
    return list(scores.sort_values(["mutation_burden", "mrna_signal"], ascending=[False, False]).index)


def _select_tcga_gdc_2025(studies: list[dict[str, Any]]) -> dict[str, Any] | None:
    for study in studies:
        name = str(_pick(study, "name", default="")).lower()
        study_id = str(_pick(study, "studyId", "study_id", default="")).lower()
        if "tcga gdc" in name and "2025" in name and "gdc" in study_id:
            return study
    return None


def _select_cptac_study(studies: list[dict[str, Any]]) -> dict[str, Any] | None:
    cptac_studies = []
    for study in studies:
        name = str(_pick(study, "name", default="")).lower()
        study_id = str(_pick(study, "studyId", "study_id", default="")).lower()
        if "cptac" not in name:
            continue
        if "gdc" in name:
            continue
        cptac_studies.append(study)
    if not cptac_studies:
        for study in studies:
            name = str(_pick(study, "name", default="")).lower()
            if "cptac" in name:
                cptac_studies.append(study)
    if not cptac_studies:
        return None
    return sorted(cptac_studies, key=_cptac_sort_key)[0]


def _cptac_sort_key(study: dict[str, Any]) -> tuple[int, int, str]:
    name = str(_pick(study, "name", default="")).lower()
    study_id = str(_pick(study, "studyId", "study_id", default="")).lower()
    match = re.search(r"(19|20)\d{2}", f"{study_id} {name}")
    year = int(match.group(0)) if match else 0
    is_gdc = 1 if "gdc" in name or study_id.endswith("_gdc") else 0
    return (is_gdc, -year, study_id)


def _format_source_footer(source: CbioPortalDataSource | None) -> str:
    if source is None:
        return ""
    parts = [
        "Source: cBioPortal",
        f"Study: {source.study.study_id}",
        f"Sample size: {len(source.sample_ids)}",
    ]
    if source.study.name:
        parts.append(f"Name: {source.study.name}")
    if source.mutation_profile is not None:
        parts.append(f"Mutation profile: {source.mutation_profile.molecular_profile_id}")
    if source.mrna_profile is not None:
        parts.append(f"Expression profile: {source.mrna_profile.molecular_profile_id}")
    if source.mutation_genes:
        parts.append(f"Mutation genes: {', '.join(source.mutation_genes)}")
    if source.mrna_genes:
        parts.append(f"Expression genes: {', '.join(source.mrna_genes)}")
    return " | ".join(parts)


def _sample_column(df: pd.DataFrame) -> str:
    for candidate in ("sampleId", "sample_id", "SAMPLE_ID"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"could not locate a sample id column in {list(df.columns)}")


def _value_column(df: pd.DataFrame) -> str:
    for candidate in ("value", "molecularValue", "molecular_value"):
        if candidate in df.columns:
            return candidate
    # Fallback to the first numeric-like column other than sampleId.
    for column in df.columns:
        if column == _sample_column(df):
            continue
        return column
    raise ValueError(f"could not locate a value column in {list(df.columns)}")


def _mutation_type_column(df: pd.DataFrame) -> str:
    for candidate in ("mutationType", "mutation_type", "variantClassification", "variant_classification"):
        if candidate in df.columns:
            return candidate
    return _value_column(df)


def _summarize_mutation_types(rows: pd.DataFrame, type_col: str) -> str:
    if rows.empty:
        return ""
    values = [str(value) for value in rows[type_col].tolist() if pd.notna(value)]
    if not values:
        return "MUT"
    unique_values = sorted(set(values))
    return ";".join(unique_values)


def _is_mutated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text not in {"0", "0.0", "False", "false", "nan", "None"}


def _pick(item: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                return item[key]
        return default
    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)
    return default
