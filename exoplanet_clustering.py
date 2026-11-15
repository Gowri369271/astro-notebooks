# =====================================================================
# Exoplanet Families: can ML rediscover planet types from raw numbers?
# Data: NASA Exoplanet Archive (pscomppars). Runs top to bottom in one
# Google Colab cell (or split it at the "# %%" markers into cells).
# =====================================================================

# %% 0. Setup ----------------------------------------------------------
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

try:
    import umap
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn"])
    try:
        import umap
    except Exception:
        umap = None  # falls back to PCA for the 2D map

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
RNG = 42

# %% 1. Download -------------------------------------------------------
cols = ["pl_name", "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt",
        "pl_orbsmax", "st_teff", "st_mass", "sy_dist", "discoverymethod"]
url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+"
       + ",".join(cols) + "+from+pscomppars&format=csv")
raw = pd.read_csv(url)
print(f"Downloaded {len(raw):,} confirmed planets")

# %% 2. Clean + feature engineering -----------------------------------
features = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "st_teff"]
df = raw.dropna(subset=features).copy()
df = df[(df[features] > 0).all(axis=1)].reset_index(drop=True)
print(f"{len(df):,} planets have all of: {features}")

X = df[features].copy()
for c in ["pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt"]:
    X[c] = np.log10(X[c])          # these span orders of magnitude
Xs = StandardScaler().fit_transform(X)

# %% 3. Pick number of clusters via silhouette ------------------------
ks = list(range(2, 11))
scores = []
for k in ks:
    labels = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit_predict(Xs)
    scores.append(silhouette_score(Xs, labels))
best_k = ks[int(np.argmax(scores))]
print(f"Best k by silhouette: {best_k} (score {max(scores):.3f})")

plt.figure(figsize=(6, 3.5))
plt.plot(ks, scores, "o-")
plt.axvline(best_k, color="red", ls="--", label=f"best k = {best_k}")
plt.xlabel("number of clusters"); plt.ylabel("silhouette score")
plt.title("How many planet families?"); plt.legend(); plt.show()

# If silhouette picks a very coarse split, use at least 4 so the map is richer
K = max(best_k, 4)
km = KMeans(n_clusters=K, n_init=20, random_state=RNG).fit(Xs)
df["cluster"] = km.labels_
print(f"Using K = {K} clusters")

# %% 4. 2D map of the planet population -------------------------------
if umap is not None:
    emb = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=RNG).fit_transform(Xs)
    method = "UMAP"
else:
    emb = PCA(n_components=2, random_state=RNG).fit_transform(Xs)
    method = "PCA"
df["x"], df["y"] = emb[:, 0], emb[:, 1]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sc = ax[0].scatter(df.x, df.y, c=df.cluster, cmap="tab10", s=10, alpha=0.8)
ax[0].set_title(f"{method} map colored by cluster")
ax[0].legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)

top_methods = df.discoverymethod.value_counts().head(5).index
for m in top_methods:
    sub = df[df.discoverymethod == m]
    ax[1].scatter(sub.x, sub.y, s=10, alpha=0.7, label=m)
ax[1].set_title(f"{method} map colored by discovery method")
ax[1].legend(fontsize=8)
for a in ax: a.set_xlabel(f"{method} 1"); a.set_ylabel(f"{method} 2")
plt.tight_layout(); plt.show()

# %% 5. What is each cluster? (profile + auto-label) -------------------
def label_cluster(r):
    rad, per = r["pl_rade"], r["pl_orbper"]
    if rad > 8:
        return "Hot Jupiter-like" if per < 10 else "Warm/cold giant"
    if rad > 4:
        return "Neptune-like"
    if rad > 1.75:
        return "Mini-Neptune"
    return "Rocky / super-Earth"

profile = df.groupby("cluster")[features].median().round(2)
profile.insert(0, "n_planets", df.cluster.value_counts().sort_index())
profile["guessed_type"] = profile.apply(label_cluster, axis=1)
print("\nMedian properties per cluster:")
display(profile)

# %% 6. Does the radius valley show up? --------------------------------
# Use every planet with a radius (not just the clustered subset), close-in only.
rv = raw.dropna(subset=["pl_rade", "pl_orbper"])
rv = rv[(rv.pl_orbper < 100) & rv.pl_rade.between(1, 4)]
plt.figure(figsize=(7, 4))
plt.hist(rv.pl_rade, bins=np.logspace(0, np.log10(4), 40), color="steelblue")
plt.xscale("log")
plt.axvspan(1.5, 2.0, color="red", alpha=0.15, label="expected valley (1.5-2 R⊕)")
plt.xlabel("planet radius (Earth radii)"); plt.ylabel("count")
plt.title(f"Radius valley check ({len(rv):,} close-in planets)")
plt.legend(); plt.show()

# %% 7. Weirdest planets via Isolation Forest --------------------------
iso = IsolationForest(contamination=0.02, random_state=RNG).fit(Xs)
df["weirdness"] = -iso.score_samples(Xs)      # higher = stranger
df["density_gcc"] = 5.51 * df.pl_bmasse / df.pl_rade**3  # Earth = 5.51 g/cc
weird = (df.sort_values("weirdness", ascending=False).head(10)
           [["pl_name", "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt",
             "density_gcc", "discoverymethod", "cluster"]].round(2))
print("\nTop 10 weirdest planets:")
display(weird)

plt.figure(figsize=(7, 5))
plt.scatter(df.pl_orbper, df.pl_rade, c="lightgray", s=8)
tw = df.sort_values("weirdness", ascending=False).head(10)
plt.scatter(tw.pl_orbper, tw.pl_rade, c="red", s=40, label="top outliers")
for _, r in tw.iterrows():
    plt.annotate(r.pl_name, (r.pl_orbper, r.pl_rade), fontsize=7, xytext=(4, 4),
                 textcoords="offset points")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("orbital period (days)"); plt.ylabel("radius (Earth radii)")
plt.title("Where the outliers live"); plt.legend(); plt.show()

# %% 8. Observation bias check ----------------------------------------
nmi = normalized_mutual_info_score(df.cluster, df.discoverymethod)
print(f"\nNormalized mutual information (cluster vs discovery method): {nmi:.2f}")
print("(0 = unrelated, 1 = the method fully determines the cluster)")

ct = pd.crosstab(df.cluster, df.discoverymethod, normalize="index")
ct = ct[ct.sum().sort_values(ascending=False).head(5).index]
ct.plot(kind="bar", stacked=True, figsize=(8, 4), colormap="tab20")
plt.ylabel("share of cluster"); plt.title("Discovery method inside each cluster")
plt.legend(fontsize=8, bbox_to_anchor=(1.01, 1)); plt.tight_layout(); plt.show()

# %% 9. The big picture: radius vs period, the classic bias plot ------
plt.figure(figsize=(8, 5.5))
sc = plt.scatter(df.pl_orbper, df.pl_rade, c=df.cluster, cmap="tab10", s=12, alpha=0.8)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("orbital period (days)"); plt.ylabel("radius (Earth radii)")
plt.title("Planet families in period-radius space")
plt.legend(*sc.legend_elements(), title="cluster", fontsize=8)
plt.show()

# %% 10. Summary -------------------------------------------------------
print("\n================ SUMMARY ================")
print(f"Planets clustered:        {len(df):,}")
print(f"Clusters found:           {K} (silhouette-best k = {best_k})")
for cid, r in profile.iterrows():
    print(f"  cluster {cid}: {int(r.n_planets):>4} planets, "
          f"median radius {r.pl_rade} R⊕, period {r.pl_orbper} d "
          f"-> {r.guessed_type}")
print(f"Cluster vs discovery NMI: {nmi:.2f}")
print(f"Strangest planet:         {weird.iloc[0].pl_name}")
print("Things to eyeball: Did the radius valley appear? Do clusters match")
print("the labels astronomers use? Is the map really just telescope bias?")
