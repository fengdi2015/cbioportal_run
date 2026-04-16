# cbio

This repo contains a workflow-oriented downloader for GEO series like `GSE150290` and a cBioPortal plotting workflow.

`GSE150290` is a GEO single-cell dataset with a processed supplementary archive:

- GEO series page: `GSE150290`
- Supplementary file: `GSE150290_RAW.tar`
- Related SRA study: `SRP261119`
- Workflow output: `downloads/GSE150290/`
- Downloaded files: `downloads/GSE150290/files/`
- Run metadata: `downloads/GSE150290/manifest.json`
- Human-readable summary: `downloads/GSE150290/report.txt`
- H5AD conversion: `geo-to-h5ad`

## Install

```powershell
pip install -e .
```

On Linux, use the same command from a shell:

```bash
python -m pip install -e .
```

If you want a plain requirements install first:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Dry run

```powershell
download-geo GSE150290 --dry-run
```

## Download

```powershell
download-geo GSE150290 --output-dir downloads
```

## Workflow graph

```powershell
download-geo GSE150290 --write-graph workflow.mmd
```

The downloader mirrors a small SRAgent-style workflow:

1. Resolve the GEO series page.
2. Summarize downloadable artifacts.
3. Download files into `files/`.
4. Write a manifest and report.

The command also accepts `--no-progress` and `--no-summaries` if you want quieter output.

## Linux

The same commands work on Linux after installation:

```bash
download-geo GSE150290 --output-dir downloads
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --sample GSM4546300_Pat01-A --output downloads/GSE150290/GSM4546300_Pat01-A.h5ad
```

## cBioPortal

Download mutation and mRNA slices from cBioPortal and draw an oncoplot-style heatmap:

```bash
cbioportal-plot \
  --study-id brca_tcga_pan_can_atlas_2018 \
  --mutation-genes TP53,KRAS,PIK3CA \
  --mrna-genes TP53,MYC,EGFR \
  --output-dir downloads/cbioportal
```

Outputs:

- `oncoplot.png`
- `relationship.png`
- `mutations.tsv`
- `mrna.tsv`
- `mrna_raw.tsv`
- `samples.txt`
- `source.json`

The cancer-family mode is deterministic: it reads exact study IDs from the frozen index file at `src/geo_downloader/cbioportal_study_index.json` by default, instead of discovering studies live at runtime.

To refresh that frozen index from the live cBioPortal API:

```bash
cbioportal-build-index --output src/geo_downloader/cbioportal_study_index.json
```

The plot is rendered as one or two stacked heatmaps depending on which modalities are available:

- mutation rows on top
- mRNA rows below
- samples are columns
If a study only has one modality, the plot is rendered for that modality alone.

For TCGA studies, the workflow selects the raw log2 TPM/FPKM-style expression profile when available and then applies a row-wise z-score transform before plotting. CPTAC studies keep their native expression profile when it is already stored as Z-scores.

For colorectal adenocarcinoma, a good test study is:

```bash
cbioportal-plot \
  --study-id coadread_tcga_pan_can_atlas_2018 \
  --mutation-genes APC,TP53,KRAS,PIK3CA,SMAD4,FBXW7 \
  --mrna-genes APC,TP53,KRAS,PIK3CA,SMAD4,FBXW7 \
  --presentation \
  --output-dir downloads/cbioportal
```

If you want the default 2025-style studies for a cancer family, use `--cancer` instead of `--study-id`. The command will run the TCGA GDC 2025 study first, and then add a CPTAC study if one exists for that cancer:

```bash
cbioportal-plot \
  --cancer luad \
  --mutation-genes STK11,KRAS,TP53 \
  --mrna-genes CD3D,CD68,FAP,CD8B,CD79A \
  --presentation \
  --output-dir downloads/cbioportal
```

If you maintain your own frozen index, point the plotter at it with `--index-file`.

The default presentation mode now writes both a heatmap and a relationship figure. The relationship figure uses raw log2 expression values for the box/violin statistics and forest summary, while the heatmap keeps the display-scale z-scores:

```bash
cbioportal-plot \
  --cancer luad \
  --mutation-genes STK11,KRAS,TP53 \
  --mrna-genes CD3D,CD68,FAP,CD8B,CD79A \
  --presentation \
  --output-dir downloads/cbioportal
```

If you want only the relationship figure, use `--figure-style relationship`; if you want only the heatmap, use `--figure-style heatmap`.

For a true live smoke test against the public cBioPortal site, run:

```bash
python scripts/smoke_cbioportal_coad.py
```

It downloads COAD data for the same driver genes, checks that mutation and mRNA tables are populated, and writes a plot plus TSV outputs under `downloads/smoke-cbioportal-coad/`.

The presentation plot version keeps samples as thin columns, orders them by mutation burden, and writes `association.tsv` with a simple mutated-vs-wild-type expression summary per gene.

For lung adenocarcinoma, a live smoke test is available for the CPTAC 2020 cohort:

```bash
smoke-cbioportal-luad-cptac
```

By default it queries `luad_cptac_2020`, uses `STK11,KRAS,TP53` as mutation genes, and writes both:

- a heatmap using the display-scale mRNA panel
- a relationship figure with violin-plus-jitter panels and a forest summary panel based on raw log2 expression values

The smoke outputs also include `mrna_raw.tsv` so the p-value and effect-size calculations remain traceable.

## Convert to H5AD

The raw GEO archive contains many sample bundles. Convert one bundle at a time:

```powershell
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --list-samples
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --sample GSM4546300_Pat01-A --output downloads/GSE150290/GSM4546300_Pat01-A.h5ad
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --all-samples --output-dir downloads/GSE150290/h5ad
```

If you omit `--sample`, the first bundle in the archive is used.
If you use `--all-samples`, the converter writes one `.h5ad` per detected GSM bundle.
