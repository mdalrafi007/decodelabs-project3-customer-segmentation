"""Stage 2: select representation + K, fit final K-Means, validate, profile. Run after segmentation.py."""
import sys, json, warnings, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
SRC = sys.argv[2]
SEED, NINIT = 42, 20
rng = np.random.default_rng(SEED)
cust = pd.read_csv(f"{OUT}/tables/customer_features.csv", index_col=0)
NUM = ["avg_quantity", "avg_unit_price", "avg_items_in_cart", "recency_days"]
X = cust[NUM]
scaler = StandardScaler().fit(X); Z = scaler.transform(X)
chk = pd.DataFrame(Z, columns=NUM).agg(["mean", "std"]).round(6)   # verify scaling
pca = PCA(random_state=SEED).fit(Z)
P3 = pca.transform(Z)[:, :3]; pca3 = PCA(3, random_state=SEED).fit(Z)
loadings = pd.DataFrame(pca.components_.T, index=NUM, columns=[f"PC{i+1}" for i in range(4)]).round(3)
R = {"scaling_check": chk.to_dict(), "loadings": loadings.to_dict(), "evr_N": pca.explained_variance_ratio_.round(4).tolist()}

# ---- candidate representations at K=2..10: N-4D vs N-PCA3 (91% var) vs N-PCA2 (67% var)
rows = []
for name, M in [("N-4D", Z), ("N-PCA3", P3), ("N-PCA2", P3[:, :2])]:
    for k in range(2, 11):
        km = KMeans(k, n_init=NINIT, random_state=SEED).fit(M); lab = km.labels_
        ref = KMeans(k, n_init=NINIT, random_state=SEED).fit_predict(Z)
        rows.append(dict(space=name, K=k, WCSS=km.inertia_, sil=silhouette_score(M, lab),
                         sil_in_4D=silhouette_score(Z, lab), ARI_vs_4D=adjusted_rand_score(ref, lab),
                         sizes=" / ".join(f"{100*s:.0f}%" for s in np.bincount(lab) / len(lab))))
cmp_ = pd.DataFrame(rows); cmp_.to_csv(f"{OUT}/tables/representation_comparison.csv", index=False)
print(cmp_[cmp_.K.isin([2, 3, 4, 5, 6])].round(3).to_string(index=False))

# ---- final model: K=4 in the 3-component PCA space (91.2% variance), validated vs 4D
K = 4
km = KMeans(K, n_init=NINIT, random_state=SEED).fit(P3)
lab0 = km.labels_
# order clusters by mean unit price (descending) for stable, readable IDs
order = pd.Series(X.avg_unit_price.values).groupby(lab0).mean().sort_values(ascending=False).index.tolist()
remap = {old: new for new, old in enumerate(order)}
lab = np.array([remap[l] for l in lab0])
cent_pca = km.cluster_centers_[order]
R.update(K=K, converged_iter=int(km.n_iter_), inertia=float(km.inertia_), sil_pca3=float(silhouette_score(P3, lab)),
         sil_4D=float(silhouette_score(Z, lab)), sizes=np.bincount(lab).tolist())

# silhouette per cluster + plot
sv = silhouette_samples(P3, lab)
R["sil_by_cluster"] = {int(c): round(float(sv[lab == c].mean()), 3) for c in range(K)}
R["neg_sil_pct"] = round(float((sv < 0).mean() * 100), 1)
fig, ax = plt.subplots(figsize=(6, 5)); y = 10; cols = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
for c in range(K):
    v = np.sort(sv[lab == c]); ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=cols[c], alpha=.8)
    ax.text(-0.08, y + len(v) / 2, f"C{c}"); y += len(v) + 10
ax.axvline(sv.mean(), ls="--", color="red"); ax.set_xlabel("silhouette coefficient"); ax.set_yticks([])
ax.set_title(f"Silhouette plot, K={K} (mean {sv.mean():.3f})"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/08_silhouette_plot_K4.png", dpi=130); plt.close(fig)

# stability: (a) 30 random seeds, (b) 50 bootstrap refits
base = lab0
seed_ari = [adjusted_rand_score(base, KMeans(K, n_init=1, random_state=s).fit_predict(P3)) for s in range(30)]
R["seed_ari_mean"], R["seed_ari_min"] = float(np.mean(seed_ari)), float(np.min(seed_ari))
boot = []
for b in range(50):
    idx = rng.choice(len(P3), len(P3), replace=True)
    boot.append(adjusted_rand_score(base, KMeans(K, n_init=10, random_state=b).fit(P3[idx]).predict(P3)))
R["boot_ari_mean"], R["boot_ari_sd"] = float(np.mean(boot)), float(np.std(boot))
# agreement of final labels with 4D model
lab4d = KMeans(K, n_init=NINIT, random_state=SEED).fit_predict(Z)
R["ari_vs_4D"] = float(adjusted_rand_score(lab, lab4d))

# ---- PCA visualisations coloured by cluster
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
raw_pca = PCA(2, random_state=SEED).fit(Z); P2 = P3[:, :2]
ev = pca.explained_variance_ratio_
for c in range(K):
    ax[0].scatter(P2[lab == c, 0], P2[lab == c, 1], s=10, alpha=.6, color=cols[c], label=f"Cluster {c}")
cp = cent_pca[:, :2]; ax[0].scatter(cp[:, 0], cp[:, 1], marker="X", s=200, c="black", edgecolor="white", label="centroids")
ax[0].set_xlabel(f"PC1 ({100*ev[0]:.1f}%)"); ax[0].set_ylabel(f"PC2 ({100*ev[1]:.1f}%)"); ax[0].legend(fontsize=8)
ax[0].set_title(f"2D PCA ({100*(ev[0]+ev[1]):.1f}% variance), coloured by K=4 clusters")
ax[1].remove(); ax3 = fig.add_subplot(1, 2, 2, projection="3d")
for c in range(K): ax3.scatter(P3[lab == c, 0], P3[lab == c, 1], P3[lab == c, 2], s=6, alpha=.5, color=cols[c])
ax3.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)"); ax3.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)"); ax3.set_zlabel(f"PC3 ({100*ev[2]:.1f}%)")
ax3.set_title(f"3D PCA ({100*ev[:3].sum():.1f}% variance)"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/09_pca_clusters_2d_3d.png", dpi=130); plt.close(fig)

# unclustered PCA scatter (required "before" view) + 30-column space view
raw = pd.read_csv(f"{OUT}/tables/customer_features.csv", index_col=0)
Zf = np.load(f"{OUT}/tables/_Zf.npy"); pf = PCA(2, random_state=SEED).fit(Zf); Pf = pf.transform(Zf)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(P2[:, 0], P2[:, 1], s=8, alpha=.5, color="grey"); ax[0].set_title("Space N, PCA before clustering")
ax[0].set_xlabel(f"PC1 ({100*ev[0]:.1f}%)"); ax[0].set_ylabel(f"PC2 ({100*ev[1]:.1f}%)")
ax[1].scatter(Pf[:, 0], Pf[:, 1], s=8, alpha=.5, color="grey"); ax[1].set_title("Space F (30 columns), PCA before clustering")
ax[1].set_xlabel(f"PC1 ({100*pf.explained_variance_ratio_[0]:.1f}%)"); ax[1].set_ylabel(f"PC2 ({100*pf.explained_variance_ratio_[1]:.1f}%)")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/10_pca_before_clustering.png", dpi=130); plt.close(fig)

# ---- profiling in original units
orig = pd.read_excel(SRC); orig["CouponCode"] = orig.CouponCode.fillna("NoCoupon")
cust["cluster"] = lab
cs = cust.copy()
cs["share_coupon_used"] = 1 - cs["coupon_NoCoupon"]
cs["share_realised"] = cs.status_Delivered + cs.status_Shipped
cs["share_cancel_return"] = cs.status_Cancelled + cs.status_Returned
cs["cart_to_qty_ratio"] = cs.avg_items_in_cart / cs.avg_quantity
prof_cols = ["avg_quantity", "avg_unit_price", "avg_items_in_cart", "recency_days", "total_spend", "cart_to_qty_ratio",
             "share_coupon_used", "share_realised", "share_cancel_return"]
overall_mean, overall_med = cs[prof_cols].mean(), cs[prof_cols].median()
tab = []
for c in range(K):
    s = cs[cs.cluster == c]
    d = {"cluster": c, "n": len(s), "pct": round(100 * len(s) / len(cs), 1)}
    for p in prof_cols:
        d[f"{p}_mean"] = s[p].mean(); d[f"{p}_median"] = s[p].median()
        d[f"{p}_idx"] = 100 * s[p].mean() / overall_mean[p]
    tab.append(d)
prof = pd.DataFrame(tab)
ov = {"cluster": "ALL", "n": len(cs), "pct": 100.0}
for p in prof_cols: ov[f"{p}_mean"] = overall_mean[p]; ov[f"{p}_median"] = overall_med[p]; ov[f"{p}_idx"] = 100
prof = pd.concat([prof, pd.DataFrame([ov])], ignore_index=True); prof.to_csv(f"{OUT}/tables/cluster_profile_numeric.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
show = prof[["cluster", "n", "pct"] + [f"{p}_mean" for p in prof_cols]].round(2)
print(show.to_string(index=False))
print(prof[["cluster"] + [f"{p}_median" for p in prof_cols]].round(2).to_string(index=False))
print(prof[["cluster"] + [f"{p}_idx" for p in prof_cols]].round(0).to_string(index=False))

# categorical composition (share of customers) per cluster, + chi-square for association (post-hoc, NOT used for clustering)
from scipy.stats import chi2_contingency
cat_rows, chi = [], {}
for col, p in {"Product": "prod", "PaymentMethod": "pay", "OrderStatus": "status", "CouponCode": "coupon", "ReferralSource": "ref"}.items():
    dc = [c for c in cs.columns if c.startswith(p + "_")]
    comp = cs.groupby("cluster")[dc].mean() * 100; comp.loc["ALL"] = cs[dc].mean() * 100
    cat_rows.append(comp.round(1))
    ct = pd.DataFrame({c: cs[cs.cluster == c][dc].sum() for c in range(K)})
    chi[col] = float(chi2_contingency(ct.values)[1])
cat_tab = pd.concat(cat_rows, axis=1); cat_tab.to_csv(f"{OUT}/tables/cluster_profile_categorical.csv")
print(cat_tab.to_string()); print("chi2 p:", {k: round(v, 4) for k, v in chi.items()})
R["chi2_p"] = chi
# one-way ANOVA/Kruskal for numerics incl. total_spend (outcome not used in clustering)
from scipy.stats import kruskal
R["kruskal_p"] = {p: float(kruskal(*[cs[cs.cluster == c][p] for c in range(K)])[1]) for p in prof_cols}
# feature contribution: eta-squared of each clustering feature
eta = {}
for f in NUM:
    gm = cs[f].mean(); ssb = sum(len(cs[cs.cluster == c]) * (cs[cs.cluster == c][f].mean() - gm) ** 2 for c in range(K))
    eta[f] = float(ssb / ((cs[f] - gm) ** 2).sum())
R["eta_sq"] = eta

# reconstructed centroids (PCA3 -> inverse PCA -> inverse scaler); approximation (drops PC4)
rec = scaler.inverse_transform(pca3.inverse_transform(cent_pca))
rec = pd.DataFrame(rec, columns=NUM).round(2); rec.insert(0, "cluster", range(K)); rec.to_csv(f"{OUT}/tables/reconstructed_centroids.csv", index=False)
print(rec.to_string(index=False))
R["centroid_pca"] = np.round(cent_pca, 3).tolist()

# profile heatmap (index vs overall)
idxcols = ["avg_quantity", "avg_unit_price", "avg_items_in_cart", "recency_days", "total_spend", "share_coupon_used", "share_realised"]
H = prof[prof.cluster != "ALL"][[f"{c}_idx" for c in idxcols]].values - 100
fig, a = plt.subplots(figsize=(9, 3.6)); im = a.imshow(H, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
a.set_xticks(range(len(idxcols))); a.set_xticklabels(idxcols, rotation=30, ha="right"); a.set_yticks(range(K)); a.set_yticklabels([f"Cluster {c}" for c in range(K)])
for i in range(K):
    for j in range(len(idxcols)): a.text(j, i, f"{H[i,j]:+.0f}%", ha="center", va="center", fontsize=8)
a.set_title("Cluster mean vs overall mean (% difference); features 1-4 drove the clustering"); fig.colorbar(im); fig.tight_layout()
fig.savefig(f"{OUT}/figures/11_profile_heatmap.png", dpi=130); plt.close(fig)

# F-space artefact check: K=7 clusters in 30-D vs Product
k7 = KMeans(7, n_init=NINIT, random_state=SEED).fit_predict(Zf)
prod = orig.groupby("CustomerID").Product.first().reindex(cust.index)
R["F_K7_vs_product_ARI"] = float(adjusted_rand_score(k7, prod))

# persist artefacts
joblib.dump({"scaler": scaler, "pca3": pca3, "kmeans": km, "label_map": remap, "features": NUM}, f"{OUT}/final_model.joblib")
out = cust.reset_index()[["CustomerID", "cluster"]]; out.to_csv(f"{OUT}/tables/customer_cluster_labels.csv", index=False)
with pd.ExcelWriter(f"{OUT}/Project3_results.xlsx") as w:
    cmp_.to_excel(w, sheet_name="representation_comparison", index=False); prof.round(3).to_excel(w, sheet_name="profile_numeric", index=False)
    cat_tab.to_excel(w, sheet_name="profile_categorical"); rec.to_excel(w, sheet_name="reconstructed_centroids", index=False)
    pd.read_csv(f"{OUT}/tables/k_sweep_all_spaces.csv").round(4).to_excel(w, sheet_name="k_sweep_all_spaces", index=False)
    cs.reset_index().round(4).to_excel(w, sheet_name="customers_with_cluster", index=False)
json.dump(R, open(f"{OUT}/tables/final_results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("loadings", "scaling_check")}, indent=1, default=float))
print(loadings)
