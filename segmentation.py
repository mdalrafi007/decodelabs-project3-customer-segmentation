"""DecodeLabs Data Science Project 3 - Customer Segmentation (PCA + K-Means).
Run: python segmentation.py <path_to_xlsx> <output_dir>
"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
warnings.filterwarnings("ignore")

SRC = sys.argv[1] if len(sys.argv) > 1 else "Dataset_for_Data_Analytics.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs"
import os; os.makedirs(f"{OUT}/figures", exist_ok=True); os.makedirs(f"{OUT}/tables", exist_ok=True)
SEED, NINIT = 42, 20
R = {}  # results collected for the report

# ---------------------------------------------------------------- 1. load + audit
raw = pd.read_excel(SRC)
df = raw.copy()
R["rows"], R["cols"] = raw.shape
R["unique_customers"] = int(df.CustomerID.nunique())
R["multi_order_customers"] = int((df.CustomerID.value_counts() > 1).sum())
R["dup_rows"] = int(df.duplicated().sum())
R["calc_mismatch"] = int((abs(df.Quantity * df.UnitPrice - df.TotalPrice) > 0.02).sum())
R["cart_lt_qty"] = int((df.ItemsInCart < df.Quantity).sum())
R["missing_coupon"] = int(df.CouponCode.isna().sum())
df["CouponCode"] = df.CouponCode.fillna("NoCoupon")  # NaN = no coupon used (structural, not an error)

# EDA figures
num = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
for a, c in zip(ax, num):
    a.hist(df[c], bins=30, color="#4C78A8", edgecolor="white"); a.set_title(c); a.set_ylabel("orders")
fig.suptitle("Numerical feature distributions (order level)"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/01_distributions.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 5, figsize=(18, 3.4))
for a, c in zip(ax, ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]):
    vc = df[c].value_counts(); a.barh(vc.index[::-1], vc.values[::-1], color="#72B7B2"); a.set_title(c)
fig.suptitle("Categorical distributions (all near-uniform)"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/02_categoricals.png", dpi=130); plt.close(fig)

c = df[num].corr()
fig, a = plt.subplots(figsize=(5, 4.2)); im = a.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels(num, rotation=45, ha="right"); a.set_yticklabels(num)
for i in range(4):
    for j in range(4): a.text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center")
a.set_title("Correlation heatmap"); fig.colorbar(im); fig.tight_layout()
fig.savefig(f"{OUT}/figures/03_correlation.png", dpi=130); plt.close(fig)
R["corr"] = c.round(3).to_dict()

# ---------------------------------------------------------------- 2. customer-level table
SNAP = df.Date.max() + pd.Timedelta(days=1)
df["recency_days"] = (SNAP - df.Date).dt.days
df["is_realised"] = df.OrderStatus.isin(["Delivered", "Shipped"]).astype(int)
g = df.groupby("CustomerID")
cust = pd.DataFrame({
    "n_orders": g.size(),
    "avg_quantity": g.Quantity.mean(),
    "avg_unit_price": g.UnitPrice.mean(),
    "avg_items_in_cart": g.ItemsInCart.mean(),
    "recency_days": g.recency_days.min(),
    "total_spend": g.TotalPrice.sum(),
})
cats = {"Product": "prod", "PaymentMethod": "pay", "OrderStatus": "status", "CouponCode": "coupon", "ReferralSource": "ref"}
dums = []
for col, p in cats.items():  # proportion of the customer's orders in each category (== one-hot for single-order customers)
    d = pd.get_dummies(df[col], prefix=p).astype(float); d["CustomerID"] = df.CustomerID
    dums.append(d.groupby("CustomerID").mean())
cat_df = pd.concat(dums, axis=1)
R["n_customers"] = len(cust)

NUM_FEATS = ["avg_quantity", "avg_unit_price", "avg_items_in_cart", "recency_days"]
Xn = cust[NUM_FEATS]                       # space N: behavioural numerics only (4-D)
Xf = pd.concat([Xn, cat_df], axis=1)       # space F: numerics + one-hot categoricals
R["dim_N"], R["dim_F"] = Xn.shape[1], Xf.shape[1]

# ---------------------------------------------------------------- 3. scale + PCA
def prep(X):
    Z = StandardScaler().fit_transform(X); return Z
Zn, Zf = prep(Xn), prep(Xf)
pca_n, pca_f = PCA(random_state=SEED).fit(Zn), PCA(random_state=SEED).fit(Zf)

def pca_summary(p, name):
    ev, cum = p.explained_variance_ratio_, np.cumsum(p.explained_variance_ratio_)
    return {"space": name, "n_comp_90": int(np.argmax(cum >= .90) + 1), "n_comp_95": int(np.argmax(cum >= .95) + 1),
            "var_2pc": float(cum[1]), "var_3pc": float(cum[2]), "evr": ev.round(4).tolist()}
R["pca_N"], R["pca_F"] = pca_summary(pca_n, "N"), pca_summary(pca_f, "F")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for p, lab, col in [(pca_f, f"Space F ({Xf.shape[1]} features)", "#E45756"), (pca_n, f"Space N ({Xn.shape[1]} features)", "#4C78A8")]:
    ev = p.explained_variance_ratio_; k = np.arange(1, len(ev) + 1)
    ax[0].plot(k, ev, "o-", ms=3, color=col, label=lab); ax[1].plot(k, np.cumsum(ev), "o-", ms=3, color=col, label=lab)
ax[0].set_title("Scree plot"); ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Explained variance ratio"); ax[0].legend()
ax[1].axhline(.95, ls="--", color="goldenrod"); ax[1].text(1, .955, "95% reference", color="goldenrod")
ax[1].axhline(.90, ls=":", color="grey"); ax[1].text(1, .905, "90%", color="grey")
ax[1].set_title("Cumulative explained variance"); ax[1].set_xlabel("Number of components"); ax[1].legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/04_scree_cumulative.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 4. K sweep on candidate spaces
def knee(ks, w):  # max perpendicular distance from the chord (kneedle-style); returns K and strength
    x, y = np.array(ks, float), np.array(w, float)
    xn, yn = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
    d = (1 - yn) - xn  # distance below chord y=1-x (curve decreasing & convex)
    return int(ks[int(np.argmax(d))]), float(d.max())

def sweep(Z, name, ks=range(2, 11)):
    rows = []
    for k in ks:
        km = KMeans(k, n_init=NINIT, random_state=SEED).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        sz = np.bincount(km.labels_) / len(Z)
        rows.append(dict(space=name, K=k, WCSS=km.inertia_, silhouette=sil, min_pct=100 * sz.min(), max_pct=100 * sz.max(),
                         sizes=";".join(f"{100*s:.1f}" for s in sz)))
    return pd.DataFrame(rows)

P2n, P3n = PCA(2, random_state=SEED).fit_transform(Zn), PCA(3, random_state=SEED).fit_transform(Zn)
P2f, P3f = PCA(2, random_state=SEED).fit_transform(Zf), PCA(3, random_state=SEED).fit_transform(Zf)
k95f = R["pca_F"]["n_comp_95"]; P95f = PCA(k95f, random_state=SEED).fit_transform(Zf)
spaces = {"N-standardised (4D)": Zn, "N-PCA2": P2n, "F-standardised (30D)": Zf, "F-PCA2": P2f, "F-PCA3": P3f, f"F-PCA95 ({k95f}D)": P95f}
sw = pd.concat([sweep(Z, n) for n, Z in spaces.items()], ignore_index=True)
sw.to_csv(f"{OUT}/tables/k_sweep_all_spaces.csv", index=False)

# permutation baseline: independently shuffle each original variable (keeps marginals, kills joint structure; one-hot blocks shuffled intact)
rng = np.random.default_rng(SEED)
def null_sil(X, ks=range(2, 11), reps=5):
    # group columns: each numeric column alone; one-hot dummies of one categorical variable shuffled together as a block
    groups = {}
    for c in X.columns:
        key = c.split("_")[0] if c.split("_")[0] in cats.values() else c
        groups.setdefault(key, []).append(c)
    out = {k: [] for k in ks}
    for _ in range(reps):
        Xp = X.copy()
        for cols in groups.values():
            Xp[cols] = X[cols].values[rng.permutation(len(X))]
        Z = StandardScaler().fit_transform(Xp)
        for k in ks: out[k].append(silhouette_score(Z, KMeans(k, n_init=5, random_state=SEED).fit_predict(Z)))
    return {k: float(np.mean(v)) for k, v in out.items()}
nullN, nullF = null_sil(Xn), null_sil(Xf)
R["null_N"], R["null_F"] = nullN, nullF

# sweep plots for the two primary spaces
def sweep_plot(names, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        s = sw[sw.space == n]; ax[0].plot(s.K, s.WCSS, "o-", label=n); ax[1].plot(s.K, s.silhouette, "o-", label=n)
    ax[0].set_title("Elbow: WCSS vs K"); ax[0].set_xlabel("K"); ax[0].set_ylabel("WCSS (inertia)"); ax[0].legend(fontsize=8)
    ax[1].set_title("Mean silhouette vs K"); ax[1].set_xlabel("K"); ax[1].set_ylabel("silhouette"); ax[1].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(f"{OUT}/figures/{fname}", dpi=130); plt.close(fig)
sweep_plot(["N-standardised (4D)", "N-PCA2"], "05_elbow_silhouette_N.png", "Space N (behavioural numerics)")
sweep_plot(["F-standardised (30D)", "F-PCA2", "F-PCA3", f"F-PCA95 ({k95f}D)"], "06_elbow_silhouette_F.png", "Space F (numerics + one-hot categoricals)")

# knees
R["knees"] = {}
for n in spaces:
    s = sw[sw.space == n]; kk, st = knee(list(s.K), list(s.WCSS)); R["knees"][n] = {"K": kk, "strength": round(st, 3)}

# null overlay figure
fig, ax = plt.subplots(figsize=(6, 4))
sN = sw[sw.space == "N-standardised (4D)"]; sF = sw[sw.space == "F-standardised (30D)"]
ax.plot(sN.K, sN.silhouette, "o-", label="N real"); ax.plot(list(nullN), list(nullN.values()), "o--", label="N shuffled-variables baseline")
ax.plot(sF.K, sF.silhouette, "s-", label="F real"); ax.plot(list(nullF), list(nullF.values()), "s--", label="F shuffled-variables baseline")
ax.set_xlabel("K"); ax.set_ylabel("mean silhouette"); ax.set_title("Real data vs. structure-free baseline"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/07_null_baseline.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 5. stability (bootstrap ARI) for every K in primary space
def stability(Z, k, B=30):
    base = KMeans(k, n_init=NINIT, random_state=SEED).fit(Z); aris = []
    for b in range(B):
        idx = rng.choice(len(Z), len(Z), replace=True)
        kb = KMeans(k, n_init=10, random_state=b).fit(Z[idx])
        aris.append(adjusted_rand_score(base.labels_, kb.predict(Z)))
    return float(np.mean(aris)), float(np.std(aris))
stab = {}
for n in ["N-standardised (4D)", "F-PCA3"]:
    stab[n] = {k: stability(spaces[n], k) for k in range(2, 9)}
R["stability"] = {n: {k: [round(a, 3), round(b, 3)] for k, (a, b) in v.items()} for n, v in stab.items()}

json.dump(R, open(f"{OUT}/tables/results.json", "w"), indent=1, default=float)
cust.assign(**{c: cat_df[c] for c in cat_df}).to_csv(f"{OUT}/tables/customer_features.csv")
np.save(f"{OUT}/tables/_Zn.npy", Zn); np.save(f"{OUT}/tables/_Zf.npy", Zf)
print(sw.drop(columns="sizes").round(3).to_string())
print(json.dumps({k: R[k] for k in ["rows", "cols", "n_customers", "multi_order_customers", "dim_N", "dim_F", "pca_N", "pca_F", "knees", "null_N", "null_F", "stability"]}, indent=1, default=float))
