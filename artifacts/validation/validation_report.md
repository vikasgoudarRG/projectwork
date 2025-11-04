# Validation Report

## Summary

Total validation checks: 6

## Pass/Fail Status

| Check | Status |
|-------|--------|
| Raw BlockID Extraction | PASS |
| Session Sequences | PASS |
| Label Distribution | PASS |
| Vocabulary Check | PASS |
| Window Readiness | PASS |
| Split Integrity | PASS |

## Statistics

### Raw Data
{
  "total_lines": 10000,
  "total_blockid_occurrences": 10000,
  "unique_blockids_raw": 1018,
  "malformed_ratio": 0.0
}

### Sequences
{
  "count_sessions": 575061,
  "len_min": 2,
  "len_max": 298,
  "len_mean": 19.433814847468355,
  "len_median": 19.0,
  "pct_lt3": 0.5129890568131034,
  "pct_gt200": 0.0076513622033140835
}

### Labels
{
  "anomalies_pct": 2.9280371995318757,
  "normals_pct": 97.07196280046813,
  "total_labels": 575061
}

### Vocabulary
{
  "vocab_size": 29,
  "min_id": 1,
  "max_id": 29,
  "dense_or_sparse": "dense",
  "problems": []
}

### Windows
{
  "total_windows_expected": 0,
  "window_file_found": false,
  "total_windows_found": 0,
  "tolerance_ok": true
}

## Final Verdict

PASS

## Repair Suggestions

No issues detected.
