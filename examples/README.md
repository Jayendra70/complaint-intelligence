# Examples

Tiny 5-row demo of what the cleaning step does, safe to open by hand.

- `example_raw.csv` — 5 rows in the exact CFPB raw format (16 columns).
- `example_processed.csv` — the 3 rows that `01_data_cleaning.ipynb` produces from it.

What happens to the 5 raw rows:

| Complaint ID | What happens | Why |
|---|---|---|
| 6690001 | kept (label 0) | normal, on-time complaint |
| 6690002 | kept (is_monetary_relief = 1) | company paid the customer |
| 6690003 | kept (is_untimely = 1) | company missed the deadline |
| 6690004 | dropped | no written narrative |
| 6690005 | dropped | duplicate of 6690001 (same text, different bureau) |

Also note the two columns that vanish in the processed file
(`Company response to consumer`, `Timely response?`) — those are removed on
purpose so the model can't peek at how the complaint ended.

Handy as a quick check: feed `example_raw.csv` through the cleaning notebook and
you should get exactly `example_processed.csv` back.
