# Clonotrace R vs Python — Real Hematopoiesis Data Report

Generated: 2026-04-03 20:12:31

## Pipeline Tests

| Test | Metric | Score | Threshold | Pass? |
|------|--------|-------|-----------|-------|
| kNN+transition | Pearson r (matched) | 0.9898 | 0.98 | PASS |
| Label spreading (prob) | Pearson r | 0.9992 | 0.95 | PASS |
| Deviance | Pearson r | 0.9476 | 0.90 | PASS |
| Clone NN dist | Pearson r (sq50) | 1.0000 | 0.95 | PASS |
| Clone NN dist | Spearman r (sq50) | 1.0000 | 0.95 | PASS |
| Clone MDS | Procrustes R² (30 dims) | 0.8647 | 0.80 | PASS |
| Leiden clustering | ARI | 0.6079 | 0.70 | FAIL |
| Clone pseudotime | |Spearman r| | 0.9661 | 0.95 | PASS |
| Profile enrichment pval | Pearson r | 0.2516 | 0.90 | FAIL |
| DEG F-stat | Spearman r | 0.7939 | 0.90 | FAIL |
| DEG Cohen's d | Spearman r | 0.9377 | 0.90 | PASS |
| SNN graph weights | Pearson r | 1.0000 | 0.98 | PASS |

## Summary

- 9/12 tests passed
- 0 warnings, 0 info items
- **Conclusion**: 3 tests failed — review discrepancies.
