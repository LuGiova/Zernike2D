# Sampling and Roughness Changes

This note summarizes only the recent changes to the complementary-plane workflow.

## Default Sampling

The `default` sampling strategy was updated to be more physically meaningful and reproducible.

Main changes:
- site-1 points are still selected sparsely at first
- if that initial subset does not produce enough pairs, extra site-1 points are taken from the discarded pool
- the extra candidates are shuffled with a deterministic seed, so the same inputs and parameters always produce the same pairs
- matching is done with Hungarian assignment on 3D physical distance
- pairs are accepted only if their physical distance is at most 6 Angstrom
- if the outer ring is too sparse, the whole ring construction is retried with a smaller outer radius
- **the representative point is now the projection of the pair midpoint onto the plane, not the segment-plane intersection**

## Roughness per Ring

The `get_complementary_plane.py` summary CSV now includes ring-wise roughness values in addition to the global roughness.

For each ring `1..10`:
- `roughness_ring{r}` is computed from the PC3 variance of the matched points in that ring
- the value is derived from the two binding sites and combined as a ring-level roughness score

## Summary CSV Structure

The final summary CSV keeps this order:
- per-ring physical columns
- per-ring Zernike columns
- global quantities such as `gyration_radius`, `flatness`, `PC3`, `roughness`, and `scalar_prod`
- `radius`
- `roughness_ring1` through `roughness_ring10`

This matches the current output structure used by the workflow.