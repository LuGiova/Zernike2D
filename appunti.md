# Zernike2D Notes

## Overview

These notes summarize the evolution of the complementary-plane workflow and the main analyses performed on the dataset.

The central idea was to compare two protein binding sites by projecting them onto a complementary plane, sampling matched interface points, and comparing geometric and Zernike-based descriptors across concentric rings.

## Development History

### 1. Initial Sampling Strategy

The first sampling mode implemented later became the `default` strategy.

At that stage:
- only single-file analysis was available
- weighted means were introduced to reduce the effect of non-uniform sampling on the plane

### 2. Additional Sampling Strategies

Two additional sampling modes were then added to improve the uniformity of the sampled distribution:

- `angular_cells`: a more geometric strategy
- `kmeans`: a more physical / data-driven strategy

For these two strategies, weighted means were not considered, because the sampling itself was already intended to reduce strong non-uniformity.

### 3. Batch Mode

Later, `get_complementary_plane` was extended with batch support.

This introduced:
- processing of `dataset.zip` containing native poses
- execution of all three sampling strategies on the batch dataset
- aggregation of results across the whole dataset

## Analyses Performed

### RDF Profiles

I plotted mean RDF curves in three different ways:

1. raw data
2. per-complex data normalized so that the maximum is 1
3. per-complex data normalized so that the minimum is 0 and the maximum is 1

### Global Quantities

I also built histograms of the global quantities and studied correlations between them using:
- Pearson correlation
- Spearman correlation

### Threshold-Based GIFs

To study how the RDF changes when applying a lower bound on the global quantities, I created GIFs where the threshold is gradually increased.

## Observations on the Runs

## 1a1u

From the circular binding-site plots, the best sampling appeared to be `angular_cells`.

Main qualitative observations:
- `kmeans` seemed to emphasize the borders of the binding site, almost as if it traced its outline
- Zernike RDFs were quite similar across the three strategies
- Zernike curves were somewhat noisy but generally increasing on average
- Physical RDFs were similar for `default` and `kmeans`
- Physical RDFs were much flatter for `angular_cells`
- For `angular_cells`, the physical RDF was almost a rising line up to ring 8, then it decreased

## dataset.zip

### Mean RDF Behavior

For the full dataset, both RDFs showed an increasing trend overall.

Some details:
- Zernike decreased only from the ninth to the tenth ring
- this last-ring drop may be due to the fact that, at large physical distances, outer regions of the binding site are being sampled, and those regions may be relatively smooth

#### Physical RDF

- `default` showed an almost exponential trend
- `kmeans` showed a very similar trend, but with a small decrease between the first and second ring
- `angular_cells` looked almost linear, with saturation in the outer rings

#### Zernike RDF

- excluding the last ring, `default` looked almost linear
- `kmeans` was also close to linear
- `angular_cells` was roughly linear in the central region
- in `angular_cells`, there was a large increase between the first and second ring, and a mild saturation between the eighth and ninth ring

### Histograms of Global Quantities

The distributions of gyration radius, roughness, and flatness were strongly skewed:
- large peak at low values
- long tails toward higher values
- the tails were especially relevant for roughness and flatness

The scalar product distribution looked approximately Gaussian, centered between `-0.2` and `-0.4`.

Interpretation:
- the normals are often oriented in opposite directions
- however, they are less antiparallel than I initially expected
- the most antiparallel results were observed with `kmeans`
- the most orthogonal results were observed with `angular_cells` and `default` weighted
- `default` normal lay in between

This is also the quantity whose mean changes the most between `default` normal and `default` weighted among all the quantities we examined, including the RDFs and the global metrics.

### Correlations Between Global Quantities

Main correlation pattern:
- gyration radius was the least correlated quantity overall
- flatness and roughness were strongly correlated
- scalar product was more correlated with roughness and flatness than with gyration radius

### Threshold GIF Analysis

The threshold that had the strongest impact on the RDF trend was the gyration radius: it was the first one to make the trend start changing, whereas the other thresholds only affected the RDFs at the very last cutoff values.

Effects of the thresholds:
- increasing the lower bound on gyration radius caused both RDFs to flatten
- the physical RDF under `angular_cells` tended to become bell-shaped
- varying roughness, flatness, or scalar product produced very similar effects
- the RDF trend was lost only at very high thresholds, where statistics became poor and only a small number of complexes remained

Strategy-specific behavior:
- the Zernike RDF tended to keep the same overall trend for all three strategies
- the physical RDF stayed stable in its overall trend for `angular_cells`
- the physical RDF tended toward a U-shape for `kmeans`
- the physical RDF tended to flatten for `default`, except for the last ring

## Main Takeaway

Among the three strategies, `angular_cells` looked like the most stable method overall.

It also seemed to provide the most geometrically balanced sampling of the binding site.

## Open Questions

Some points still need interpretation:
- why, among all the quantities we examined, the mean of the scalar product histogram changes the most between `default` normal and `default` weighted
- why `kmeans` emphasizes borders so clearly
- why scalar product is distributed around moderately negative values rather than values close to `-1`
- how much of the last-ring Zernike drop is a true geometric effect versus a sampling artifact

## Suggested Next Step

A useful next step would be to turn these observations into a short results section with:
- a comparison table between the three strategies
- a summary of the global quantity correlations
- a short discussion of why `angular_cells` appears more stable
