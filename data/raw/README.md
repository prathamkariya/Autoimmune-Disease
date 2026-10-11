# Raw data

The notebooks read `data/raw/dataset.xlsx`, which is **not stored in this repository**.

* **Source:** Harvard Dataverse, *Rheumatic and autoimmune diseases dataset*, DOI
  [10.7910/DVN/VM4OR3](https://doi.org/10.7910/DVN/VM4OR3). Check the dataset page for its terms of use.
* **Expected file:** `dataset.xlsx` (rename the downloaded file if necessary), a single sheet with 12,085 rows and 15
  columns: Age, Gender, ESR, CRP, RF, Anti-CCP, HLA-B27, ANA, Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm, C3, C4, Disease.
* **SHA-256 of the file used for all results:**
  `499ec1507c9ae3604a13cef55d2c5be02310e73604e438abceb17634467bc246`

Verify with `sha256sum data/raw/dataset.xlsx` (Linux/macOS) or `Get-FileHash data\raw\dataset.xlsx` (Windows).
Notebook 02 also asserts the shape `(12085, 15)` and stops with a clear message if the file is missing.

`data/processed/` is created by notebook 02 and is regenerated, so it is not stored either.
