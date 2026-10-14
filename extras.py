"""Stage 3: data-quality table, skewness, missing chart, elbow w/ knee marker, silhouette plots K=2..5,
sensitivity analyses (drop ItemsInCart, log-transform), realised-revenue profile, GMM comparison.
Run after segmentation.py and final_model.py:  python extras.py <out_dir> <xlsx>"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from scipy.stats import skew
warnings.filterwarnings("ignore")
OUT, SRC = sys.argv[1], sys.argv[2]; SEED, NINIT = 42, 20
raw = pd.read_excel(SRC)
cust = pd.read_csv(f"{OUT}/tables/customer_features.csv", index_col=0)
lab = pd.read_csv(f"{OUT}/tables/customer_cluster_labels.csv", index_col=0).cluster.reindex(cust.index).values
NUM = ["avg_quantity", "avg_unit_price", "avg_items_in_cart", "recency_days"]
Z = StandardScaler().fit_transform(cust[NUM]); P3 = PCA(3, random_state=SEED).fit_transform(Z)
R = {}

# ---- skewness + missing-value chart
sk = raw[["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]].apply(skew).round(3)
sk["recency_days (customer level)"] = round(float(skew(cust.recency_days)), 3)
sk.to_csv(f"{OUT}/tables/skewness.csv", header=["skew"]); R["skew"] = sk.to_dict()
miss = raw.isna().mean() * 100
fig, a = plt.subplots(figsize=(7, 3.6)); a.barh(miss.index[::-1], miss.values[::-1], color="#E45756")
a.set_xlabel("% missing"); a.set_title("Missing values by column (only CouponCode = 25.75%)"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/12_missing_values.png", dpi=130); plt.close(fig)

# ---- elbow with knee marker + silhouette in final space; detailed silhouette plots K=2..5
ks = list(range(2, 11)); w = [KMeans(k, n_init=NINIT, random_state=SEED).fit(P3).inertia_ for k in ks]
s = [silhouette_score(P3, KMeans(k, n_init=NINIT, random_state=SEED).fit_predict(P3)) for k in ks]
x = np.array(ks, float); yn = (np.array(w) - min(w)) / (max(w) - min(w)); d = (1 - yn) - (x - 2) / 8; kn = ks[int(d.argmax())]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, w, "o-"); ax[0].plot([2, 10], [w[0], w[-1]], ":", color="grey", label="chord")
ax[0].axvline(kn, color="red", ls="--", label=f"max chord distance K={kn} (weak, strength {d.max():.2f})"); ax[0].legend(fontsize=8)
ax[0].set_title("Elbow, final PCA-3 space"); ax[0].set_xlabel("K"); ax[0].set_ylabel("WCSS")
ax[1].plot(ks, s, "o-", color="green"); ax[1].axvline(4, color="red", ls="--", label="selected K=4"); ax[1].legend(fontsize=8)
ax[1].set_title("Mean silhouette, final PCA-3 space"); ax[1].set_xlabel("K"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/13_elbow_silhouette_final_space.png", dpi=130); plt.close(fig)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, k in zip(ax, [2, 3, 4, 5]):
    l = KMeans(k, n_init=NINIT, random_state=SEED).fit_predict(P3); sv = silhouette_samples(P3, l); y = 10
    for c in range(k):
        v = np.sort(sv[l == c]); a.fill_betweenx(np.arange(y, y + len(v)), 0, v, alpha=.8); y += len(v) + 10
    a.axvline(sv.mean(), ls="--", color="red"); a.set_title(f"K={k}, mean {sv.mean():.3f}, min cluster {min(sv[l==c].mean() for c in range(k)):.3f}"); a.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/figures/14_silhouette_plots_K2to5.png", dpi=130); plt.close(fig)

# ---- unclustered 3D PCA
fig = plt.figure(figsize=(6, 5)); a = fig.add_subplot(projection="3d"); a.scatter(*P3.T, s=5, alpha=.4, color="grey")
a.set_xlabel("PC1 (41.4%)"); a.set_ylabel("PC2 (25.2%)"); a.set_zlabel("PC3 (24.6%)"); a.set_title("3D PCA before clustering (91.2%)")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/15_pca3d_before_clustering.png", dpi=130); plt.close(fig)

# ---- sensitivity analyses (agreement with final labels, K=4)
def fit(M): return KMeans(4, n_init=NINIT, random_state=SEED).fit_predict(M)
sens = []
def add(name, M):
    l = fit(M); sens.append(dict(variant=name, silhouette=round(silhouette_score(M, l), 3), ARI_vs_final=round(adjusted_rand_score(lab, l), 3),
                                 sizes=" / ".join(f"{100*v:.0f}%" for v in np.bincount(l) / len(l))))
add("Final (4 features, PCA-3)", P3)
F3 = ["avg_quantity", "avg_unit_price", "recency_days"]; Zs = StandardScaler().fit_transform(cust[F3]); add("Drop items_in_cart (3 features, standardised)", Zs)
Zl = StandardScaler().fit_transform(np.log1p(cust[NUM])); add("log1p-transform then standardise (4 features)", Zl)
Zr = (cust[NUM] - cust[NUM].median()) / (cust[NUM].quantile(.75) - cust[NUM].quantile(.25)); add("Robust scaling (median/IQR)", Zr.values)
add("Raw unscaled features (demonstrates scale distortion)", cust[NUM].values)
sens = pd.DataFrame(sens); sens.to_csv(f"{OUT}/tables/sensitivity.csv", index=False); print(sens.to_string(index=False))
raw_l = fit(cust[NUM].values); R["unscaled_cluster_spread_by_recency_corr"] = float(np.corrcoef(raw_l, cust.recency_days)[0, 1])

# ---- GMM alternative (optional comparison)
gm = GaussianMixture(4, n_init=10, random_state=SEED).fit(P3); gl = gm.predict(P3)
R["gmm_k4_ARI_vs_kmeans"] = float(adjusted_rand_score(lab, gl)); R["gmm_bic"] = {k: float(GaussianMixture(k, n_init=5, random_state=SEED).fit(P3).bic(P3)) for k in range(1, 9)}

# ---- realised-revenue profile (Delivered/Shipped only)
o = raw.merge(pd.Series(lab, index=cust.index, name="cluster"), left_on="CustomerID", right_index=True)
o["real"] = o.OrderStatus.isin(["Delivered", "Shipped"])
rv = o.groupby("cluster").apply(lambda g: pd.Series({"orders": len(g), "gross_order_value": g.TotalPrice.sum(), "realised_value": g.TotalPrice[g.real].sum(),
                                                       "realised_share_of_value_%": 100 * g.TotalPrice[g.real].sum() / g.TotalPrice.sum(),
                                                       "realised_order_rate_%": 100 * g.real.mean(), "AOV_all": g.TotalPrice.mean()}))
rv.loc["ALL"] = [len(o), o.TotalPrice.sum(), o.TotalPrice[o.real].sum(), 100 * o.TotalPrice[o.real].sum() / o.TotalPrice.sum(), 100 * o.real.mean(), o.TotalPrice.mean()]
rv["share_of_total_realised_%"] = 100 * rv.realised_value / rv.loc["ALL", "realised_value"]; rv.round(1).to_csv(f"{OUT}/tables/realised_revenue_by_cluster.csv")
print(rv.round(1).to_string())
json.dump(R, open(f"{OUT}/tables/extras_results.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
with pd.ExcelWriter(f"{OUT}/Project3_results.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    sens.to_excel(w, sheet_name="sensitivity", index=False); rv.round(1).to_excel(w, sheet_name="realised_revenue")
