# MLflow runs — DistilBERT + LoRA, Tier 1 (Colab)

The transformer runs were tracked in MLflow on Colab. The raw tracking store
(`mlruns_colab/`) is kept local, not on GitHub; it was in the first commit
(`940f85a`) and removed from the repo in `a74f77b`. This table restates its four
runs (experiment id `136298397859160813`). The Phase 2 baselines were tracked in a
separate local store (`mlruns/`, experiment `phase2_baselines_6mo`).

All four runs: `distilbert-base-uncased`, LoRA, learning rate 2e-4,
`max_length` 256. Validation is on the natural-mix validation set.

| Run name | Task | Loss | Validation metric | Selected |
|---|---|---|---|---|
| `product_plain_ce_v2` | Product (9 classes) | plain cross-entropy | macro-F1 0.7473 | yes |
| `product_focal_v2` | Product (9 classes) | focal | macro-F1 0.7302 | no |
| `monetary_focal_v2` | Monetary relief | focal | PR-AUC 0.3548 | yes |
| `monetary_weighted_ce_v2` | Monetary relief | weighted cross-entropy | PR-AUC 0.3519 | no |

The selected runs scored 0.7386 macro-F1 (product) and 0.3308 PR-AUC (monetary)
on the 34,694-row test set (`reports/transformer_report.md`).
