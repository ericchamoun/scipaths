# SciPaths Data

This directory contains the public claim-level SciPaths release.

## Files

| File | Split | Records | Labels |
| --- | --- | ---: | --- |
| `training.json` | silver training | 2,444 | target contributions, enabling contributions, and groundings |
| `dev.json` | gold development | 50 | target contributions, enabling contributions, and groundings |
| `test.json` | public test inputs | 212 | target contributions only |

The test split is intentionally blind: it includes only `idx` and `target_contribution`.

## Labeled Split Schema

`training.json` and `dev.json` are JSON lists. Each row has:

- `claim_id`: globally unique claim identifier.
- `claim_idx`: zero-based target-contribution index within the source paper.
- `conference_year`: publication venue year when available.
- `target_paper_id`: source paper identifier.
- `target_title`: source paper title.
- `target_contribution`: the target contribution to decompose.
- `downstream_usage_evidence`: downstream citation evidence used to derive or support the target contribution.
- `enabling_contributions`: gold or silver enabling contributions for the target contribution.

Each enabling contribution has:

- `enabling_contribution_id`: unique ID within the split.
- `enabling_contribution`: normalized enabling contribution text.
- `canonical_ref_id`: selected primary grounding reference, or `__NONE__`.
- `canonical_grounding`: metadata for the primary grounding, or `null`.
- `additional_ref_ids`: IDs for additional grounding studies.
- `additional_groundings`: metadata for additional grounding studies.
- `canonical_annotation`: `roles`, `contribution`, `rationale`, and `evidence_span`.

## Test Split Schema

`test.json` is a JSON list with:

- `idx`: integer test example ID.
- `target_contribution`: the target contribution to decompose.

## Notes

`training.json` is silver data and should be treated as training material, not as a gold evaluation split. `dev.json` is the public gold development split. `test.json` is the public test input split.

## License Note

The annotation data in this directory may include metadata and short paper-derived evidence spans. Underlying source materials retain their own licenses and terms. Users are responsible for complying with applicable upstream conditions when reusing or redistributing paper-derived content.
