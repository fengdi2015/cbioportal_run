# GEO Downloader

This repo contains a workflow-oriented downloader for GEO series like `GSE150290`.

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

## Convert to H5AD

The raw GEO archive contains many sample bundles. Convert one bundle at a time:

```powershell
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --list-samples
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --sample GSM4546300_Pat01-A --output downloads/GSE150290/GSM4546300_Pat01-A.h5ad
geo-to-h5ad downloads/GSE150290/files/GSE150290_RAW.tar --all-samples --output-dir downloads/GSE150290/h5ad
```

If you omit `--sample`, the first bundle in the archive is used.
If you use `--all-samples`, the converter writes one `.h5ad` per detected GSM bundle.
