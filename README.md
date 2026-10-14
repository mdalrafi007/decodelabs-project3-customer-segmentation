# DecodeLabs Data Science Project 3: Customer Segmentation (PCA + K-Means)

**Author:** Al Rafi | **Data:** `Dataset_for_Data_Analytics.xlsx` (1,200 e-commerce orders, Jan 2023 to Jun 2025) | **Seed:** 42, `n_init=20`

## 1. Headline findings (read this first)

1. **The data is order-level, not customer-level, and has 14 columns, not 20+.** I aggregated to 1,189 customers (1,178 have exactly one order) and report dimensionality honestly. I did not invent features.
2. **The data has very weak natural cluster structure.** Silhouette scores are low (about 0.23 in the full 4-feature space) and only slightly above a structure-free baseline built by shuffling each variable (about 0.20). The 2D/3D PCA clouds are continuous, not separated blobs (Figure 9). The four clusters below are best read as **a practical partition of a continuum**, not discovered natural groups.
3. **Final model:** K-Means, K = 4, fitted on a 3-component PCA of 4 standardised behavioural features (91.2% variance retained). The four segments differ by basket size, price level and recency (Sections 7 and 9).
4. **The segmentation is sensitive to preprocessing choices** (Section 8): dropping one correlated feature, log-transforming or robust-scaling changes the labels substantially (ARI vs final 0.27 to 0.74). Treat the segments as one reasonable view, not a stable truth.
5. **Categorical fields (payment, status, coupon, referral) do not differ between clusters** (all p > 0.05), so no channel or coupon claims are made.

## 2. Requirements checklist

| # | Requirement | Where satisfied | Status |
|---|---|---|---|
| 1 | PCA on 20+ columns to 2-3 dims | Space F: 30 columns (4 numeric + 26 one-hot) → Fig 4, 6, 10; Space N: 4 features → Fig 4, 9 | **Partly.** Raw data has 14 columns; 30 only after one-hot encoding. 2 PCs keep 10.8% of F variance, 3 keep 15.7% |
| 2 | Elbow Method | Fig 5, 6; Table 4 | Done. No sharp elbow (stated honestly) |
| 3 | Silhouette Score | Fig 5, 6, 7, 8; Table 4 | Done |
| 4 | Evidence-based K | Section 6 | Done, with caveats |
| 5 | Business personas | Section 9 | Done |
| 6 | Elbow knee marker, detailed silhouette plots, 2D/3D before and after clustering | Fig 13, 14, 15, 9, 10 | Done |
| 7 | Standardisation before distance methods | `StandardScaler`, verified mean 0 / std 1 | Done |

## 3. Data audit

| Check | Result |
|---|---|
| Shape | 1,200 rows × 14 columns |
| Unit of observation | Order. 1,189 unique CustomerIDs; 11 customers have 2 orders |
| Duplicates | 0 duplicated rows, 0 duplicated OrderID, 0 duplicated TrackingNumber |
| Missing values | Only `CouponCode`: 309 rows (25.75%). Treated as **"NoCoupon"** (no coupon used), not an error |
| Consistency | `Quantity × UnitPrice = TotalPrice` in all 1,200 rows; `ItemsInCart ≥ Quantity` in all rows |
| Identifiers / unsuitable | OrderID, TrackingNumber, CustomerID, ShippingAddress (655 values, all "NNN Main St", uninformative) excluded from distances |
| Redundant | `TotalPrice` is exactly Quantity × UnitPrice, so excluded from clustering and used only for profiling |
| Outliers | None removed. The max order (3,456.40) is a valid large order |
| Oddities | Every product spans the same price range ($11 to $700); order status is almost uniform (only 39% Delivered or Shipped; Cancelled 21%, Returned 21%, Pending 20%). This pattern is typical of synthetic data |
| Correlations | Quantity vs ItemsInCart 0.65; UnitPrice vs TotalPrice 0.72; Quantity vs TotalPrice 0.62 |

Skewness: Quantity 0.03, UnitPrice −0.03, ItemsInCart 0.00, recency −0.05 (all near-symmetric, so no log transform was needed); TotalPrice 0.89 (right-skewed, profiling only, so medians are reported alongside means).

**Data-quality log**

| Issue | Severity | Action | Rationale |
|---|---|---|---|
| Order-level rows, not customers | High | Aggregated to 1,189 customers | Segmentation targets customers; 11 repeat customers (0.9%) are averaged |
| CouponCode missing (309 rows, 25.75%) | Low | Filled with "NoCoupon" | Blank means no coupon used; no rows dropped |
| Fewer than 20 raw columns | Medium | Reported honestly; one-hot space (30 columns) used only for the PCA demonstration | No invented features |
| TotalPrice = Quantity × UnitPrice (exact) | Medium | Excluded from clustering | Would triple-count the same information |
| ItemsInCart correlated with Quantity (r = 0.65) | Medium | Kept, tested in sensitivity (Section 8) | Real behavioural signal, but double-weights basket size |
| ShippingAddress uninformative / personal | Low | Excluded | 655 values, all "NNN Main St" |
| IDs (Order, Tracking, Customer) | n/a | Excluded from distances; CustomerID kept to map labels | Identifiers carry no behaviour |
| Outliers | None found | No rows removed | Max order 3,456.40 is valid (5 × about $691) |
| 61% of orders not Delivered or Shipped | High (for revenue claims) | Kept in clustering; realised revenue reported separately (Section 8) | Status is not a customer trait; dropping would remove 61% of data |

Figures: `figures/01_distributions.png`, `02_categoricals.png`, `03_correlation.png`, `12_missing_values.png`.

## 4. Feature preparation

- **Customer-level table:** `avg_quantity`, `avg_unit_price`, `avg_items_in_cart`, `recency_days` (days from last order to 2025-07-01), `total_spend` (profiling only), `n_orders` (dropped: 99% equal to 1, near-constant).
- **Space N (primary):** the 4 behavioural features above, standardised: z = (x − mean) / std. Scaling check: all means 0.0, all std 1.0.
- **Space F (comparison):** Space N plus category proportions for Product (7), Payment (5), Status (5), Coupon (4), Referral (5) = **30 columns** (rank 25, because each group of dummies sums to 1).
- **Caveat on one-hot + Euclidean distance:** dummy variables are binary and near-uniform here, so they add distance without adding behavioural signal. Section 5 shows this causes artefacts.

## 5. PCA results

| Space | Variance in 2 PCs | in 3 PCs | PCs for 90% | PCs for 95% |
|---|---|---|---|---|
| N (4 features) | **66.6%** | **91.2%** | 3 | 4 |
| F (30 columns) | 10.8% | 15.7% | 22 | 23 |

- **Space F cannot be compressed to 2-3 dimensions without losing about 85% of the variance.** Its variance is spread almost evenly (PC1 5.8%, PC2 5.0%, PC3 4.9%), which means little correlated structure for PCA to exploit. The "4 or 5 components keep 95%" picture in the course slides does not apply to this dataset.
- **Space F silhouette is an artefact.** In 30-D, K = 7 gives silhouette 0.135, which equals the shuffled-variable baseline (0.135), and the 7 clusters match `Product` almost perfectly (ARI = 0.99). K-Means is just recovering the 7 product categories. I therefore did **not** use Space F for the final model.
- Loadings (Space N): PC1 = basket size (quantity 0.70, items in cart 0.70); PC2 = price plus recency (0.79, 0.60); PC3 = recency versus price (0.79, −0.61).

Figures: `04_scree_cumulative.png` (scree + cumulative with 95% line), `10_pca_before_clustering.png` (2D, both spaces), `15_pca3d_before_clustering.png` (3D).

## 6. Choosing the representation and K

**Representation.** I compared clustering in standardised 4-D, PCA-3 (91.2%) and PCA-2 (66.6%):

| Representation | K=4 silhouette (own space) | K=4 silhouette measured in 4-D | Agreement with 4-D labels (ARI) |
|---|---|---|---|
| 4-D standardised | 0.233 | 0.233 | 1.000 |
| **PCA-3 (chosen)** | 0.270 | 0.233 | **0.977** |
| PCA-2 | 0.352 | 0.197 | 0.440 |

PCA-2 looks better (0.352) only because discarding dimensions makes points appear closer; measured back in 4-D it is worse (0.197) and its labels disagree with the full-information solution. PCA-3 reproduces the 4-D labels (ARI 0.977), satisfies the 2-3 dimension requirement and keeps 91.2% of the variance.

**K sweep (PCA-3 space, K = 2 to 10).** Selected rows:

| K | WCSS | Silhouette | Cluster sizes |
|---|---|---|---|
| 2 | 2,922.6 | 0.288 | 50% / 50% |
| 3 | 2,394.6 | 0.256 | 30% / 40% / 31% |
| **4** | **1,940.5** | **0.270** | 24% / 23% / 26% / 27% |
| 5 | 1,659.2 | 0.268 | 19% / 21% / 18% / 22% / 21% |
| 6 | 1,422.6 | 0.275 | 17% / 17% / 13% / 18% / 16% / 19% |

Full table for all spaces: `tables/k_sweep_all_spaces.csv`, `tables/representation_comparison.csv`.

**Evidence, stated honestly:**
- **Elbow:** WCSS falls smoothly (per-step drops 18%, 19%, 14.5%, 14%, 11.5% ...). The chord-distance knee method gives K = 5 in 4-D and PCA-3 (strength about 0.26, weak) and K = 4 in PCA-2. **There is no sharp elbow.**
- **Silhouette:** nearly flat across K (0.256 to 0.288 in PCA-3). The highest is K = 2 (0.288), but K = 2 is the coarsest split (essentially low versus high basket size).
- **Baseline:** real silhouettes are only modestly above shuffled-variable data (4-D: K=4 0.233 vs 0.196; K=2 0.266 vs 0.187), and no better by K = 8 (0.225 vs 0.219). Structure is weak.
- **Stability (bootstrap ARI, 4-D):** K=2 0.98, K=4 0.78, K=5 0.67, K=8 0.76.
- **Choice:** K = 4 because it has (a) a silhouette bump over K = 3 and K = 5, (b) better stability than K = 5, (c) balanced segments (23% to 27%), (d) knee agreement in PCA-2, and (e) enough segments to be actionable. **K = 2 is statistically cleaner but not useful for targeting; K = 4 versus K = 5 is close to a tie.** I would not defend K = 4 as "the true number of clusters".

Figures: `05_elbow_silhouette_N.png`, `06_elbow_silhouette_F.png`, `13_elbow_silhouette_final_space.png` (knee marked), `07_null_baseline.png`, `08_silhouette_plot_K4.png`, `14_silhouette_plots_K2to5.png`.

## 7. Final model: K = 4 (PCA-3 space)

- Converged in 11 iterations; inertia 1,940.5; sizes 290 / 305 / 317 / 277.
- Silhouette 0.270 (PCA-3) and 0.233 (4-D). Per cluster: 0.273, 0.273, 0.259, 0.276. Only 0.3% of customers have negative silhouette.
- **Stability:** across 30 random single initialisations ARI averages 0.72 (minimum 0.25, so some single-start runs hit poor local optima; `n_init=20` avoids this). Bootstrap refits: ARI 0.73 ± 0.20. **Treat assignments of individual customers near boundaries as uncertain.**
- **Features driving separation** (eta² = share of each feature's variance explained by cluster): avg_quantity 0.61, avg_items_in_cart 0.53, recency 0.45, avg_unit_price 0.43. Note quantity and items-in-cart are correlated (0.65), so basket size carries roughly double weight (limitation).
- Artefacts saved: `final_model.joblib` (scaler, PCA, K-Means, label map), `tables/customer_cluster_labels.csv`.

Figures: `09_pca_clusters_2d_3d.png`, `11_profile_heatmap.png`.

### Cluster profiles (original units; means, with index vs overall = 100)

| Cluster | n (%) | Avg quantity | Avg unit price | Items in cart | Recency (days) | Total spend (mean / median) |
|---|---|---|---|---|---|---|
| 0 | 290 (24.4%) | 4.12 (140) | $533 (149) | 7.2 (131) | 545 (118) | $2,195 / $2,113 (206) |
| 1 | 305 (25.7%) | 1.87 (64) | $406 (114) | 3.8 (69) | 210 (45) | $783 / $641 (74) |
| 2 | 317 (26.7%) | 1.93 (65) | $307 (86) | 4.0 (73) | 682 (147) | $596 / $490 (56) |
| 3 | 277 (23.3%) | 4.07 (138) | $175 (49) | 7.3 (132) | 406 (88) | $724 / $636 (61) |
| All | 1,189 | 2.95 | $357 | 5.5 | 463 | $1,064 / $834 |

Reconstructed centroids (PCA-3 → inverse PCA → inverse scaler, an approximation that drops PC4) are in `tables/reconstructed_centroids.csv` and match the observed means within about 1%. Total spend differs by construction (it is quantity × price), so it is a consequence of the clustering features, not independent validation.

**Not different across clusters** (post-hoc tests, not used for clustering): coupon use (p = 0.91), share of Delivered or Shipped orders (p = 0.48), cancel/return share (p = 0.72), order-status mix (p = 0.96), referral source (p = 0.75), payment method (p = 0.052). Product mix differs mildly (p = 0.014), which does not survive a Bonferroni correction across 5 tests (threshold 0.01).

## 8. Robustness checks and realised revenue (`extras.py`)

**Sensitivity (K = 4, agreement with the final labels):**

| Variant | Silhouette | ARI vs final |
|---|---|---|
| Final (4 features, PCA-3) | 0.270 | 1.00 |
| Drop ItemsInCart (3 features) | 0.281 | 0.42 |
| log1p then standardise | 0.300 | 0.27 |
| Robust scaling (median/IQR) | 0.215 | 0.74 |
| Unscaled raw features | 0.392 | 0.29 |

- The labels move a lot when preprocessing changes. This is another sign of a continuum rather than natural groups. The segments are best described by the broad pattern (basket size × price × recency), not by exact membership.
- Unscaled features give the highest silhouette (0.392) only because recency (range about 900 days) dominates the distance. A higher silhouette is not a better segmentation here, which shows why scaling is required.
- **Alternative method:** a 4-component Gaussian mixture on the same PCA-3 space agrees only partly with K-Means (ARI 0.43); BIC is lowest at 4 to 5 components (10,633 at 4, 10,630 at 5), loosely consistent with the chosen K.

**Realised revenue by segment (Delivered or Shipped orders only):**

| Cluster | Orders | Gross order value | Realised value | Realised order rate | Share of total realised value |
|---|---|---|---|---|---|
| 0 | 292 | $636,555 | $237,873 | 37.3% | **48.7%** |
| 1 | 309 | $238,674 | $95,981 | 42.1% | 19.6% |
| 2 | 319 | $189,005 | $71,286 | 36.4% | 14.6% |
| 3 | 280 | $200,527 | $83,621 | 39.6% | 17.1% |
| All | 1,200 | $1,264,762 | $488,760 | 38.8% | 100% |

Cluster 0 holds 24% of customers but about 49% of realised value, so it remains the most valuable segment on a realised basis. Only about 39% of order value overall is realised, with similar rates across clusters.

## 9. Persona matrix

| Cluster | Persona | Evidence | Commercial value | Proposed strategy (hypothesis) | KPI to track | Caveat |
|---|---|---|---|---|---|---|
| 0 | **Bulk Premium Buyers** | Highest quantity (4.1), price ($533) and spend (2.06× average); 24% of customers | Highest order value in this data | Bundle and volume offers on premium items; account-style service | Realised-order rate (Delivered or Shipped), average order value | Only 37% of this segment's orders are Delivered or Shipped; it still yields about 49% of realised value |
| 1 | **Recent Mid-Ticket Light Buyers** | Smallest baskets (1.9 items), mid-high price ($406), **most recent** (210 days, 45% of average) | Active base to grow | Cross-sell and basket-building (for example "add a second item") | Items per order, 90-day repeat rate | Recency is the date of a single order; it does not prove loyalty |
| 2 | **Lapsed Light Buyers** (potential lapse risk) | Small baskets, lower price ($307), **least recent** (682 days), lowest spend | Low current value; reactivation target | Low-cost win-back test versus a holdout | Reactivation rate (any new order within 90 days of contact) | "Lapse risk" rests on recency alone; most customers have only one order, so there is no purchase history to confirm churn |
| 3 | **Bulk Budget Buyers** | High quantity (4.1) at the lowest price ($175, 49% of average) | Volume at low ticket | Multi-buy or bundle pricing on low-price items | Units per order, margin per order | No margin or cost data, so profitability is unknown |

**Needs, barriers and channel (hypotheses, not measured):**

| Cluster | Possible need / barrier | Channel |
|---|---|---|
| 0 Bulk Premium | Fulfilment reliability on large orders (63% of their orders are not Delivered or Shipped) | Account-style outreach; no channel preference found in the data |
| 1 Recent Mid-Ticket | A reason to add a second item | Post-purchase follow-up while recency is fresh |
| 2 Lapsed Light | Low price and low basket; may simply have bought once | Low-cost reactivation test with a holdout group |
| 3 Bulk Budget | Price-driven volume | Multi-buy bundles; margin must be checked first |

No segment-specific channel recommendation is made, because referral source, payment method and coupon use do not differ between clusters in this data. Suggested actions are hypotheses to A/B test, not conclusions.

## 10. Limitations

- Weak cluster structure: silhouette 0.23 to 0.27, close to structure-free baselines, and a continuous PCA cloud. The clusters are a convenient partition, not definitive customer types.
- Almost every customer has one order, so "customer behaviour" is really "single-order behaviour". No loyalty, frequency or lifetime value can be measured.
- 61% of orders are Cancelled, Returned or Pending; spend counts all orders. A realised-revenue version would change the value ranking and is a recommended next step.
- Quantity and ItemsInCart overlap (r = 0.65); dropping ItemsInCart changes the labels materially (ARI 0.42, Section 8).
- Results change with scaling and transformation choices (ARI 0.27 to 0.74), so individual customer membership should not be over-interpreted.
- K = 4 versus K = 5 is a near tie; stability is moderate (bootstrap ARI 0.73 ± 0.20).
- Descriptive and non-causal. The dataset has the statistical look of synthetic data (uniform categories, product-independent prices), so conclusions may not transfer to real retail data.

## 11. Figure index

01 distributions · 02 categoricals · 03 correlation · 04 scree and cumulative variance (95% line) · 05/06 elbow and silhouette by space · 07 structure-free baseline · 08 silhouette plot K=4 · 09 PCA 2D and 3D coloured by cluster · 10/15 PCA before clustering · 11 profile heatmap · 12 missing values · 13 elbow with knee marker, final space · 14 silhouette plots K=2 to 5.

## 12. Reproduce

**Notebook:** `DecodeLabs_Project3_Customer_Segmentation.ipynb` runs the whole pipeline end to end with outputs and charts included (it was executed here by a script that runs each cell and stores its real output). Deterministic results (PCA, K sweep, final model, profiles) match this README exactly. Resampling-based figures (bootstrap stability, shuffled baseline) use fewer repetitions in the notebook, so they differ slightly (for example K=4 bootstrap ARI 0.75 vs 0.78 here). The scripts below remain the reference numbers.

```bash
python segmentation.py Dataset_for_Data_Analytics.xlsx .   # audit, EDA, PCA, K sweep, baselines
python final_model.py . Dataset_for_Data_Analytics.xlsx    # final model, profiles, tables, Excel workbook
python extras.py . Dataset_for_Data_Analytics.xlsx         # sensitivity, realised revenue, GMM, extra figures
```
Requires numpy, pandas, scikit-learn 1.8, matplotlib, scipy, openpyxl, joblib. All three scripts were executed to produce every number and chart above.
