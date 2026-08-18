# Drift report

- Current data: `test.parquet` (34694 rows)
- Reference rows (training time): 47121
- **Drift detected: True**

| Check | Value | Verdict |
|---|---|---|
| narrative_length_psi | PSI=0.012 | none |
| narrative_length_ks | KS=0.0338 (p=9.2e-05) | drifted |
| predicted_product_psi | PSI=0.7942 | major |

PSI guide: <0.10 none, 0.10-0.25 moderate, >0.25 major. KS: p<0.05 means the length distributions differ.
