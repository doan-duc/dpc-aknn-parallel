# Theoretical Foundation of DPC-AKNN

## 1. Motivation

DPC-AKNN improves upon the original DPC in two main aspects:
- The density function no longer relies on the entire dataset or a static cutoff threshold, but is constrained to the k-nearest neighbors (kNN).
- The cluster assignment stage does not use a single-step label propagation (which is prone to error accumulation). Instead, it utilizes an association matrix, iterative updates, and subsequent refinement using kNN voting.

## 2. Local Density Function

For each point `x_i`, the local density is calculated as:

`rho_i = sum_{x_j in N^k_{x_i}} exp(-(d_ij^2) / (d_c^2))`

Interpretation:
- Only the `k`-nearest neighbors are considered, reducing noise from distant points.
- The Gaussian kernel shape is preserved to leverage distance information, which performs better than a binary cutoff kernel.

## 3. Global Threshold `d_c`

First, compute a local threshold for each point:

`d_ci = mean(knn_dist_i) + std(knn_dist_i)`  according to Eq. (6)

Then, derive the global values:
- `d_bar_c = mean(d_ci)` according to Eq. (8)
- `sigma_c = std(d_ci, ddof=1)` according to Eq. (9)
- `d_c = d_bar_c + sigma_c` according to Eq. (7)

This design allows `d_c` to adapt to local data distributions while remaining a consistent global threshold used throughout the algorithm.

## 4. Cluster Center Selection

The distance to the nearest point with higher density is defined as:

`delta_i = min_{j: rho_j > rho_i} d_ij`

For the point with the highest local density:

`delta_i = max_j d_ij`

Then, the decision metric is computed:

`gamma_i = rho_i * delta_i`

The `n_c` points with the largest `gamma` values are selected as the cluster centers.

## 5. Initial Cluster Construction

Each cluster center `c_j` initializes a cluster `C_j` containing:
- The center `c_j` itself.
- All `k`-nearest neighbors of the center.

Next, clusters are expanded based on the admission rules. For a point `x_p` already in a cluster, we consider a point `x_q` in `x_p`'s kNN. `x_q` is admitted if and only if:
1. `x_q` is not yet labeled.
2. `d_pq <= mean distance from x_p to its kNN`.
3. `distance(centroid_j, x_q) <= d_c`.

Condition 2 uses the mean distance, not the maximum. Condition 3 uses the global threshold `d_c`.

## 6. Assigning Low-Density Points via Association Matrix

For the set of unassigned points `U`, construct the association matrix:

`A(i, r) = sum_{x_l in N^k_{x_i} intersect C_r} (1 / d_il) * rho_l * rho_i`

Then, iteratively perform the following:
1. Find the maximum element `(i*, r*)`.
2. Assign `x_{i*}` to cluster `r*`.
3. Remove row `i*`.
4. Update column `r*` according to Eq. (11) for unassigned points that have `x_{i*}` in their kNN.

This step dominates the time complexity of the entire algorithm.

## 7. Error Correction (Refinement)

After all points have been tentatively assigned:
- Sort the points in descending order of local density `rho`.
- For each point, count label votes within its `k`-nearest neighbors.
- Reassign the point to the cluster with the majority of votes.
- In case of a tie, choose the cluster with the smallest average distance to the neighbors belonging to that cluster.

## 8. Final Assignment of Remaining Points

If there are still unlabeled points, for each cluster `r`, calculate:

`zeta_ir = mean distance from x_i to its k-nearest neighbors in cluster r`

Then assign:

`r* = argmin_r zeta_ir`

## 9. Computational Complexity

According to the paper, the primary bottleneck is the iterative association matrix update loop, leading to an overall complexity of:

`O(n_c * n^2)`

This high complexity is the primary motivation for introducing parallelized implementations:
- `cpu_parallel/` implemented using OpenMP.
- `gpu_parallel/` implemented using CUDA.
