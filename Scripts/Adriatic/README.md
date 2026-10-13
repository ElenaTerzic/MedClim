# Adriatic

Ingestion and harmonization of the regional Adriatic in situ datasets used in the
MedClim multiplatform analysis.

## Directory structure

- `preprocessing/` – ingestion of the Adriatic source datasets and export to long-format CSV

## Preprocessing

- `read_CTD_19_23.py`
- `read_IOR_Adriatic_database_1987_2019_big.py`
- `read_IOR_Palagruza_1952_2018.py`
- `read_IRB_Northern_Adriatic_Transect_1979_2018.py`
- `diagnostics/fix_adriatic_outliers_per_basin.py` – run after the four readers, on their
  long-format tables placed in `Data/Adriatic/Processed/`: reassigns casts tagged
  `BASIN_3ADR == 'out'` to DJ1/DJ2/DJ3 by nearest centroid and writes the `*_long_fixed.csv`
  tables (new column `BASIN_3ADR_FIXED`) that the MultiPlatform stages read.

These scripts ingest the source XLSX/netCDF datasets and export harmonized long-format CSV
files for downstream use. The Palagruža, RB Northern Adriatic, and IOR readers assign
Adriatic basin labels during processing.

They are the provenance for the four cleaned Adriatic datasets deposited in the
accompanying Zenodo data record:

| script | deposited dataset |
|---|---|
| `read_CTD_19_23.py` | `adriatic_ctd_mfsd_2019_2023_clean.csv` |
| `read_IOR_Adriatic_database_1987_2019_big.py` | `adriatic_ior_database_1987_2019_clean.csv` |
| `read_IOR_Palagruza_1952_2018.py` | `adriatic_palagruza_transect_1952_2018_clean.csv` |
| `read_IRB_Northern_Adriatic_Transect_1979_2018.py` | `adriatic_rb_northern_transect_1979_2018_clean.csv` |

## Notes

- Processed data are stored outside the repository.
- The SSL (salinity stratification layer) workflows, the Adriatic-specific figure scripts,
  and the basin-polygon diagnostic previously kept here belong to a separate line of work
  and have been moved out of this repository.
