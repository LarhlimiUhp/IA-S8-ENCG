"""
============================================================
ALGORITHMES DE PREDICTION SUPERVISEE -- Partie 3
============================================================
Algorithmes couverts :

  CLASSIFICATION :
    1.  Ridge Classifier
    2.  Naive Bayes Multinomial (texte)
    3.  Naive Bayes Bernoulli
    4.  Perceptron
    5.  Passive-Aggressive Classifier
    6.  Calibration de modeles (CalibratedClassifierCV)

  REGRESSION :
    7.  Regression Polynomiale
    8.  Regression Quantile
    9.  Regression Isotonique
    10. RANSAC Regressor (robuste)
    11. Theil-Sen Regressor (robuste)
    12. Tweedie Regressor (GLM generalise)

  SYNTHESE GLOBALE :
    13. Tableau comparatif de TOUS les algorithmes
    14. Guide de selection algorithmique
============================================================
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              mean_squared_error, r2_score,
                              mean_absolute_error, brier_score_loss,
                              calibration_curve)

# ============================================================
# DONNEES
# ============================================================
iris     = datasets.load_iris()
breast   = datasets.load_breast_cancer()
wine     = datasets.load_wine()
diabetes = datasets.load_diabetes()
california = datasets.fetch_california_housing()

# Classification splits
X_iris_tr, X_iris_te, y_iris_tr, y_iris_te = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(
    breast.data, breast.target, test_size=0.2, random_state=42, stratify=breast.target)

# Regression splits
X_dia_tr, X_dia_te, y_dia_tr, y_dia_te = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42)
X_cal_tr, X_cal_te, y_cal_tr, y_cal_te = train_test_split(
    california.data, california.target, test_size=0.2, random_state=42)

sc = StandardScaler()
X_iris_tr_s = sc.fit_transform(X_iris_tr); X_iris_te_s = sc.transform(X_iris_te)
sc2 = StandardScaler()
X_bc_tr_s   = sc2.fit_transform(X_bc_tr);  X_bc_te_s   = sc2.transform(X_bc_te)
sc3 = StandardScaler()
X_dia_tr_s  = sc3.fit_transform(X_dia_tr); X_dia_te_s  = sc3.transform(X_dia_te)
sc4 = StandardScaler()
X_cal_tr_s  = sc4.fit_transform(X_cal_tr); X_cal_te_s  = sc4.transform(X_cal_te)

def plot_cm(y_true, y_pred, names, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Prediction"); ax.set_ylabel("Reel")

def plot_reg(y_true, y_pred, title, ax):
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color="#3498db")
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    r2 = r2_score(y_true, y_pred)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Reel"); ax.set_ylabel("Predit")
    ax.text(0.05, 0.92, f"R2={r2:.3f}", transform=ax.transAxes,
            fontsize=9, color="#e74c3c", fontweight="bold")

def reg_info(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:35s} | RMSE={rmse:.3f} | R2={r2:.4f}")
    return r2

# ============================================================
# SECTION A : CLASSIFICATION
# ============================================================
print("=" * 65)
print("CLASSIFICATION -- Partie 3")
print("=" * 65)

clf_res3 = {}

# ============================================================
# 1. RIDGE CLASSIFIER
# ============================================================
print("\n" + "=" * 65)
print("1. RIDGE CLASSIFIER")
print("=" * 65)

from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

# Ridge Classifier : convertit la classification en regression Ridge
# Classe predite = argmax des sorties de regression
rc = Pipeline([("sc", StandardScaler()),
               ("rc", RidgeClassifier(alpha=1.0))])
rc.fit(X_iris_tr, y_iris_tr)
pred_rc = rc.predict(X_iris_te)
acc_rc  = accuracy_score(y_iris_te, pred_rc)
clf_res3["RidgeClassifier"] = acc_rc
print(f"[Iris] RidgeClassifier | Accuracy : {acc_rc:.4f}")

# RidgeClassifierCV : selection automatique de alpha par CV
rcv = Pipeline([("sc", StandardScaler()),
                ("rcv", RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0]))])
rcv.fit(X_iris_tr, y_iris_tr)
acc_rcv = accuracy_score(y_iris_te, rcv.predict(X_iris_te))
print(f"[Iris] RidgeClassifierCV alpha optimal | Accuracy : {acc_rcv:.4f}")

# Comparaison des alphas
alphas = np.logspace(-3, 3, 20)
accs_alpha = []
for a in alphas:
    pipe_a = Pipeline([("sc", StandardScaler()),
                       ("rc", RidgeClassifier(alpha=a))])
    pipe_a.fit(X_iris_tr, y_iris_tr)
    accs_alpha.append(accuracy_score(y_iris_te, pipe_a.predict(X_iris_te)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_iris_te, pred_rc, iris.target_names, "Ridge Classifier -- Iris", axes[0])
axes[1].semilogx(alphas, accs_alpha, "bo-", lw=2)
axes[1].set_xlabel("Alpha (regularisation)"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Ridge Classifier : Accuracy vs Alpha -- Iris", fontweight="bold")
axes[1].grid(True)
plt.suptitle("RIDGE CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 2. NAIVE BAYES MULTINOMIAL -- Simulation donnees comptage
# ============================================================
print("\n" + "=" * 65)
print("2. NAIVE BAYES MULTINOMIAL")
print("=" * 65)

from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Chargement d'un sous-ensemble de 4 categories de 20 Newsgroups
categories = ["sci.med", "sci.space", "comp.graphics", "rec.sport.hockey"]
train_data = fetch_20newsgroups(subset="train", categories=categories,
                                 remove=("headers","footers","quotes"))
test_data  = fetch_20newsgroups(subset="test",  categories=categories,
                                 remove=("headers","footers","quotes"))

# Pipeline TF-IDF + MultinomialNB (classifieur texte classique)
mnb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                               min_df=2, sublinear_tf=True)),
    ("mnb",   MultinomialNB(alpha=0.1))   # alpha = lissage de Laplace
])
mnb_pipe.fit(train_data.data, train_data.target)
pred_mnb = mnb_pipe.predict(test_data.data)
acc_mnb  = accuracy_score(test_data.target, pred_mnb)
clf_res3["MultinomialNB (text)"] = acc_mnb
print(f"[20 Newsgroups - 4 categories] TF-IDF + MultinomialNB | Accuracy : {acc_mnb:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(test_data.target, pred_mnb, categories,
        "MultinomialNB -- 20 Newsgroups", axes[0])

# Top features (mots les plus discriminants) par classe
feat_names = np.array(mnb_pipe["tfidf"].get_feature_names_out())
log_probs   = mnb_pipe["mnb"].feature_log_prob_  # (n_classes, n_features)
for i, cat in enumerate(categories[:2]):          # afficher 2 classes
    top_idx = np.argsort(log_probs[i])[-10:]
    axes[1].barh([f"{cat[:15]}: {feat_names[j]}" for j in top_idx],
                  log_probs[i][top_idx], alpha=0.7)
axes[1].set_title("Top features -- MultinomialNB", fontweight="bold")
axes[1].set_xlabel("Log probabilite")
plt.suptitle("NAIVE BAYES MULTINOMIAL (classification de texte)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 3. NAIVE BAYES BERNOULLI
# ============================================================
print("\n" + "=" * 65)
print("3. NAIVE BAYES BERNOULLI")
print("=" * 65)

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer

# BernoulliNB : features BINAIRES (0/1)
# Seuil de binarisation (0 = absent, 1 = present)
bnb_pipe = Pipeline([
    ("bin", Binarizer(threshold=0.0)),      # binarise : > 0 -> 1, sinon 0
    ("bnb", BernoulliNB(alpha=1.0))
])
bnb_pipe.fit(X_bc_tr_s, y_bc_tr)
pred_bnb = bnb_pipe.predict(X_bc_te_s)
acc_bnb  = accuracy_score(y_bc_te, pred_bnb)
clf_res3["BernoulliNB"] = acc_bnb
print(f"[Breast Cancer] BernoulliNB | Accuracy : {acc_bnb:.4f}")

# Comparaison Gaussian vs Bernoulli NB
from sklearn.naive_bayes import GaussianNB
gnb_bc = GaussianNB().fit(X_bc_tr_s, y_bc_tr)
acc_gnb = accuracy_score(y_bc_te, gnb_bc.predict(X_bc_te_s))
print(f"[Breast Cancer] GaussianNB  | Accuracy : {acc_gnb:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_bc_te, pred_bnb, breast.target_names,
        "BernoulliNB -- Breast Cancer", axes[0])
axes[1].bar(["GaussianNB", "BernoulliNB"], [acc_gnb, acc_bnb],
             color=["#3498db", "#2ecc71"])
axes[1].set_ylim(0.85, 1.0); axes[1].set_ylabel("Accuracy")
axes[1].set_title("GaussianNB vs BernoulliNB -- Breast Cancer", fontweight="bold")
for x, v in enumerate([acc_gnb, acc_bnb]):
    axes[1].text(x, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")
plt.suptitle("NAIVE BAYES BERNOULLI", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 4. PERCEPTRON
# ============================================================
print("\n" + "=" * 65)
print("4. PERCEPTRON")
print("=" * 65)

from sklearn.linear_model import Perceptron

perc = Pipeline([("sc",   StandardScaler()),
                 ("perc", Perceptron(max_iter=1000, random_state=42,
                                     tol=1e-4, eta0=0.1))])
perc.fit(X_iris_tr, y_iris_tr)
pred_perc = perc.predict(X_iris_te)
acc_perc  = accuracy_score(y_iris_te, pred_perc)
clf_res3["Perceptron"] = acc_perc
print(f"[Iris] Perceptron | Accuracy : {acc_perc:.4f}")

# Convergence : erreur par epoch (via partial_fit)
perc_raw = Perceptron(max_iter=1, warm_start=True, random_state=42)
errs_perc = []
for epoch in range(50):
    sc_p = StandardScaler().fit(X_iris_tr)
    perc_raw.fit(sc_p.transform(X_iris_tr), y_iris_tr)
    errs_perc.append(1 - accuracy_score(y_iris_te,
                      perc_raw.predict(sc_p.transform(X_iris_te))))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_iris_te, pred_perc, iris.target_names, "Perceptron -- Iris", axes[0])
axes[1].plot(errs_perc, "r-", lw=2)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Taux d'erreur")
axes[1].set_title("Convergence Perceptron -- Iris", fontweight="bold")
axes[1].grid(True)
plt.suptitle("PERCEPTRON", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 5. PASSIVE-AGGRESSIVE CLASSIFIER
# ============================================================
print("\n" + "=" * 65)
print("5. PASSIVE-AGGRESSIVE CLASSIFIER")
print("=" * 65)

from sklearn.linear_model import PassiveAggressiveClassifier

pag = Pipeline([("sc",  StandardScaler()),
                ("pag", PassiveAggressiveClassifier(C=1.0, max_iter=1000,
                                                     random_state=42))])
pag.fit(X_bc_tr, y_bc_tr)
pred_pag = pag.predict(X_bc_te)
acc_pag  = accuracy_score(y_bc_te, pred_pag)
clf_res3["PassiveAggressive"] = acc_pag
print(f"[Breast Cancer] PA Classifier | Accuracy : {acc_pag:.4f}")

# Effet du parametre C
Cs = np.logspace(-3, 2, 15)
accs_C = []
for C_val in Cs:
    p = Pipeline([("sc", StandardScaler()),
                  ("pa", PassiveAggressiveClassifier(C=C_val, max_iter=1000, random_state=42))])
    p.fit(X_bc_tr, y_bc_tr)
    accs_C.append(accuracy_score(y_bc_te, p.predict(X_bc_te)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_bc_te, pred_pag, breast.target_names,
        "Passive-Aggressive -- Breast Cancer", axes[0])
axes[1].semilogx(Cs, accs_C, "bo-", lw=2)
axes[1].set_xlabel("C (agressivite)"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("PA Classifier : Accuracy vs C -- Breast Cancer", fontweight="bold")
axes[1].grid(True)
plt.suptitle("PASSIVE-AGGRESSIVE CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 6. CALIBRATION DE MODELES
# ============================================================
print("\n" + "=" * 65)
print("6. CALIBRATION DE MODELES (CalibratedClassifierCV)")
print("=" * 65)

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Modeles de base sur Breast Cancer (binaire)
base_models = {
    "SVM (non calibre)": SVC(kernel="rbf", C=1.0),
    "RF (non calibre)" : RandomForestClassifier(n_estimators=100, random_state=42),
}
calibrated_models = {
    "SVM + Platt (sigmoid)" : CalibratedClassifierCV(SVC(kernel="rbf"), cv=5, method="sigmoid"),
    "SVM + Isotonic"        : CalibratedClassifierCV(SVC(kernel="rbf"), cv=5, method="isotonic"),
    "RF + Isotonic"         : CalibratedClassifierCV(
                                  RandomForestClassifier(n_estimators=100, random_state=42),
                                  cv=5, method="isotonic"),
}

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")

for name, model in {**base_models, **calibrated_models}.items():
    sc_cal = StandardScaler()
    Xtr = sc_cal.fit_transform(X_bc_tr); Xte = sc_cal.transform(X_bc_te)
    model.fit(Xtr, y_bc_tr)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xte)[:, 1]
    else:
        prob = (model.decision_function(Xte) - model.decision_function(Xte).min()) / \
               (model.decision_function(Xte).max() - model.decision_function(Xte).min())
    frac_pos, mean_pred = calibration_curve(y_bc_te, prob, n_bins=10)
    brier = brier_score_loss(y_bc_te, prob)
    acc   = accuracy_score(y_bc_te, model.predict(Xte))
    print(f"  {name:35s} | Brier={brier:.4f} | Acc={acc:.4f}")
    ax.plot(mean_pred, frac_pos, "o-", lw=1.5, label=f"{name} (B={brier:.3f})")

ax.set_xlabel("Probabilite predite moyenne"); ax.set_ylabel("Fraction de positifs reels")
ax.set_title("Courbes de calibration -- Breast Cancer (binaire)", fontweight="bold")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True)
plt.suptitle("CALIBRATION DE MODELES", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# SECTION B : REGRESSION
# ============================================================
print("\n" + "=" * 65)
print("REGRESSION -- Partie 3")
print("=" * 65)

reg_res3 = {}

# ============================================================
# 7. REGRESSION POLYNOMIALE
# ============================================================
print("\n" + "=" * 65)
print("7. REGRESSION POLYNOMIALE")
print("=" * 65)

from sklearn.linear_model import LinearRegression, Ridge

# Sur Diabetes : comparer degres 1, 2, 3 + regularisation Ridge
degrees = [1, 2, 3]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, deg in zip(axes, degrees):
    poly_pipe = Pipeline([
        ("poly",  PolynomialFeatures(degree=deg, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0) if deg > 1 else LinearRegression())
    ])
    poly_pipe.fit(X_dia_tr, y_dia_tr)
    pred_poly = poly_pipe.predict(X_dia_te)
    r2_poly   = r2_score(y_dia_te, pred_poly)
    reg_res3[f"Poly(deg={deg})"] = r2_poly
    n_feats = poly_pipe["poly"].n_output_features_
    print(f"  Degre {deg} | Features: {n_feats:4d} | R2={r2_poly:.4f} | "
          f"RMSE={np.sqrt(mean_squared_error(y_dia_te, pred_poly)):.2f}")
    plot_reg(y_dia_te, pred_poly, f"Poly deg={deg} -- Diabetes", ax)

plt.suptitle("REGRESSION POLYNOMIALE -- Diabetes", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 8. REGRESSION QUANTILE
# ============================================================
print("\n" + "=" * 65)
print("8. REGRESSION QUANTILE")
print("=" * 65)

from sklearn.linear_model import QuantileRegressor

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
preds_q   = {}
for q in quantiles:
    qr = Pipeline([("sc", StandardScaler()),
                   ("qr", QuantileRegressor(quantile=q, alpha=0.001,
                                            solver="highs"))])
    qr.fit(X_dia_tr, y_dia_tr)
    preds_q[q] = qr.predict(X_dia_te)

q50_r2 = r2_score(y_dia_te, preds_q[0.5])
reg_res3["QuantileReg (Q50)"] = q50_r2
print(f"  Regression Quantile Q50 | R2={q50_r2:.4f}")

# Visualisation : intervalle de prediction [Q10, Q90]
sorted_idx = np.argsort(y_dia_te)
ys = y_dia_te[sorted_idx]; x_p = np.arange(len(ys))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(x_p, ys, "k-", lw=1.5, label="Valeur reelle")
axes[0].plot(x_p, preds_q[0.5][sorted_idx], "b--", lw=1.5, label="Q50 (mediane)")
axes[0].fill_between(x_p, preds_q[0.1][sorted_idx], preds_q[0.9][sorted_idx],
                      alpha=0.25, color="blue", label="Interval [Q10, Q90]")
axes[0].fill_between(x_p, preds_q[0.25][sorted_idx], preds_q[0.75][sorted_idx],
                      alpha=0.35, color="green", label="Interval [Q25, Q75]")
axes[0].set_title("Regression Quantile -- Diabetes", fontweight="bold")
axes[0].set_xlabel("Echantillon (trie par y)"); axes[0].legend(fontsize=8)

# Coverage : % de vraies valeurs dans l'intervalle [Q10, Q90]
in_interval = ((y_dia_te >= preds_q[0.1]) & (y_dia_te <= preds_q[0.9])).mean()
axes[1].bar(["Coverage [Q10,Q90]", "Attendu"], [in_interval, 0.80],
             color=["#3498db", "#95a5a6"])
axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("Proportion")
axes[1].set_title(f"Coverage : {in_interval:.1%} (attendu ~80%)", fontweight="bold")
for x, v in enumerate([in_interval, 0.80]):
    axes[1].text(x, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.suptitle("REGRESSION QUANTILE", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 9. REGRESSION ISOTONIQUE
# ============================================================
print("\n" + "=" * 65)
print("9. REGRESSION ISOTONIQUE")
print("=" * 65)

from sklearn.isotonic import IsotonicRegression

# Isotonic : contraint la fonction a etre MONOTONE croissante
# Utile pour des relations ordonnees (calibration, dose-effet)
# Fonctionne sur 1 feature uniquement -> utiliser 1ere composante PCA

from sklearn.decomposition import PCA
pca1 = PCA(n_components=1)
X_dia_1d_tr = pca1.fit_transform(X_dia_tr_s).ravel()
X_dia_1d_te = pca1.transform(X_dia_te_s).ravel()

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(X_dia_1d_tr, y_dia_tr)
pred_iso = iso.predict(X_dia_1d_te)
r2_iso   = r2_score(y_dia_te, pred_iso)
reg_res3["Isotonic"] = r2_iso
print(f"  Isotonic Regression (1 feature PCA) | R2={r2_iso:.4f}")

# Comparaison : regression lineaire 1D vs isotonique
from sklearn.linear_model import LinearRegression
lr1d = LinearRegression()
lr1d.fit(X_dia_1d_tr.reshape(-1,1), y_dia_tr)
pred_lr1d = lr1d.predict(X_dia_1d_te.reshape(-1,1))

x_sort = np.sort(X_dia_1d_te)
x_s_idx = np.argsort(X_dia_1d_te)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_dia_1d_te, y_dia_te, alpha=0.3, s=15, label="Donnees")
axes[0].plot(x_sort, iso.predict(x_sort), "r-", lw=2.5,
              label="Isotonique (monotone)")
axes[0].plot(x_sort, lr1d.predict(x_sort.reshape(-1,1)), "g--",
              lw=2, label="Lineaire")
axes[0].set_title("Isotonic vs Lineaire -- Diabetes (PC1)", fontweight="bold")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("Progression diabete")
axes[0].legend()

axes[1].bar(["Lineaire 1D", "Isotonique"],
             [r2_score(y_dia_te, pred_lr1d), r2_iso],
             color=["#2ecc71", "#e74c3c"])
axes[1].set_ylabel("R2"); axes[1].set_ylim(0, 0.6)
axes[1].set_title("R2 : Lineaire vs Isotonique", fontweight="bold")
plt.suptitle("REGRESSION ISOTONIQUE", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 10. RANSAC REGRESSOR (robuste aux outliers)
# ============================================================
print("\n" + "=" * 65)
print("10. RANSAC REGRESSOR")
print("=" * 65)

from sklearn.linear_model import RANSACRegressor

# RANSAC : Random Sample Consensus -- robuste aux outliers extremes
ransac = Pipeline([
    ("sc",     StandardScaler()),
    ("ransac", RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.5,         # 50% minimum d'inliers
        residual_threshold=50,   # seuil pour considerer un point comme outlier
        random_state=42,
        max_trials=200
    ))
])
ransac.fit(X_dia_tr, y_dia_tr)
pred_ransac = ransac.predict(X_dia_te)
r2_ransac   = r2_score(y_dia_te, pred_ransac)
reg_res3["RANSAC"] = r2_ransac
print(f"  RANSAC | R2={r2_ransac:.4f}")

# Inliers detectes par RANSAC
inlier_mask = ransac["ransac"].inlier_mask_
n_inliers    = inlier_mask.sum()
n_outliers   = (~inlier_mask).sum()
print(f"  Inliers train : {n_inliers}/{len(y_dia_tr)} | Outliers : {n_outliers}")

# Comparaison OLS vs RANSAC avec outliers
y_noisy = y_dia_tr.copy()
idx_out = np.random.default_rng(42).choice(len(y_noisy), 30, replace=False)
y_noisy[idx_out] += 250
sc_r = StandardScaler()
Xtr_s = sc_r.fit_transform(X_dia_tr); Xte_s = sc_r.transform(X_dia_te)
ols_n  = LinearRegression().fit(Xtr_s, y_noisy)
ran_n  = RANSACRegressor(residual_threshold=60, random_state=42).fit(Xtr_s, y_noisy)
r2_ols_n = r2_score(y_dia_te, ols_n.predict(Xte_s))
r2_ran_n = r2_score(y_dia_te, ran_n.predict(Xte_s))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg(y_dia_te, pred_ransac, f"RANSAC -- Diabetes (R2={r2_ransac:.3f})", axes[0])
axes[1].bar(["OLS (avec outliers)", "RANSAC (avec outliers)"],
             [r2_ols_n, r2_ran_n], color=["#e74c3c", "#2ecc71"])
axes[1].set_ylabel("R2"); axes[1].set_ylim(0, 0.6)
axes[1].set_title(f"OLS vs RANSAC en presence de 30 outliers", fontweight="bold")
for x, v in enumerate([r2_ols_n, r2_ran_n]):
    axes[1].text(x, max(v, 0) + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
plt.suptitle("RANSAC REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 11. THEIL-SEN REGRESSOR (robuste)
# ============================================================
print("\n" + "=" * 65)
print("11. THEIL-SEN REGRESSOR")
print("=" * 65)

from sklearn.linear_model import TheilSenRegressor

ts = Pipeline([("sc", StandardScaler()),
               ("ts", TheilSenRegressor(random_state=42, max_subpopulation=1000))])
ts.fit(X_dia_tr, y_dia_tr)
pred_ts = ts.predict(X_dia_te)
r2_ts   = r2_score(y_dia_te, pred_ts)
reg_res3["TheilSen"] = r2_ts
print(f"  Theil-Sen | R2={r2_ts:.4f}")

# Comparaison robustesse : OLS vs Huber vs TheilSen vs RANSAC avec outliers
from sklearn.linear_model import HuberRegressor
methods = {
    "OLS"      : LinearRegression(),
    "Huber"    : HuberRegressor(epsilon=1.35),
    "TheilSen" : TheilSenRegressor(random_state=42, max_subpopulation=500),
    "RANSAC"   : RANSACRegressor(residual_threshold=60, random_state=42),
}
print("\n  Comparaison robustesse (avec 30 outliers sur Diabetes) :")
r2s_rob = []
for name, m in methods.items():
    m.fit(Xtr_s, y_noisy)
    r2_m = r2_score(y_dia_te, m.predict(Xte_s))
    r2s_rob.append(r2_m)
    print(f"    {name:12s} | R2={r2_m:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg(y_dia_te, pred_ts, f"Theil-Sen -- Diabetes (R2={r2_ts:.3f})", axes[0])
axes[1].bar(list(methods.keys()), r2s_rob,
             color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])
axes[1].set_ylabel("R2 (avec outliers)"); axes[1].set_ylim(0, 0.6)
axes[1].set_title("Regresseurs robustes vs OLS -- Diabetes + outliers", fontweight="bold")
for x, v in enumerate(r2s_rob):
    axes[1].text(x, max(v,0) + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.suptitle("THEIL-SEN REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 12. TWEEDIE REGRESSOR (GLM Generalise)
# ============================================================
print("\n" + "=" * 65)
print("12. TWEEDIE REGRESSOR (GLM)")
print("=" * 65)

from sklearn.linear_model import TweedieRegressor

# Distribution Tweedie : generalise Gaussian (p=0), Poisson (p=1), Gamma (p=2)
tweedie_params = [
    ("Gaussian (p=0)", 0),
    ("Poisson  (p=1)", 1),
    ("Gamma    (p=2)", 2),
]

# California Housing : prix positifs -> bien adapte au Tweedie
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, power) in zip(axes, tweedie_params):
    link_fn = "identity" if power == 0 else "log"
    tw = Pipeline([
        ("sc", StandardScaler()),
        ("tw", TweedieRegressor(power=power, alpha=0.5,
                                 link=link_fn, max_iter=1000))
    ])
    # Clip negatifs pour Poisson/Gamma (exigent y > 0)
    y_tr_pos = np.clip(y_cal_tr, 0.01, None)

    tw.fit(X_cal_tr_s, y_tr_pos)
    pred_tw = tw.predict(X_cal_te_s)
    r2_tw   = r2_score(y_cal_te, pred_tw)
    reg_res3[f"Tweedie({name[:3]})"] = r2_tw
    print(f"  GLM {name:20s} | R2={r2_tw:.4f} | "
          f"RMSE={np.sqrt(mean_squared_error(y_cal_te, pred_tw)):.4f}")
    plot_reg(y_cal_te, pred_tw, f"Tweedie {name} -- California", ax)

plt.suptitle("TWEEDIE REGRESSOR (GLM generalise) -- California Housing",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# SYNTHESE GLOBALE TOUTES LES PARTIES
# ============================================================
print("\n" + "=" * 65)
print("SYNTHESE GLOBALE -- TOUS LES ALGORITHMES")
print("=" * 65)

# --- Classification : comparaison recapitulative ---
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                   SGDClassifier, Perceptron,
                                   PassiveAggressiveClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               BaggingClassifier, VotingClassifier,
                               StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                            QuadraticDiscriminantAnalysis)

all_clfs = {
    "Logistic Regression" : Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
    "Ridge Classifier"    : Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier(alpha=1.0))]),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)"           : Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", random_state=42))]),
    "KNN (k=5)"           : Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(5))]),
    "Naive Bayes"         : GaussianNB(),
    "AdaBoost"            : AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42),
    "MLP (256-128)"       : Pipeline([("sc", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42))]),
    "Extra Trees"         : ExtraTreesClassifier(n_estimators=100, random_state=42),
    "LDA"                 : LinearDiscriminantAnalysis(),
    "QDA"                 : QuadraticDiscriminantAnalysis(),
    "SGD Classifier"      : Pipeline([("sc", StandardScaler()), ("m", SGDClassifier(loss="log_loss", random_state=42))]),
    "Bagging"             : BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42, n_jobs=-1),
    "Perceptron"          : Pipeline([("sc", StandardScaler()), ("m", Perceptron(max_iter=1000, random_state=42))]),
    "PassiveAggressive"   : Pipeline([("sc", StandardScaler()), ("m", PassiveAggressiveClassifier(max_iter=1000, random_state=42))]),
    "BernoulliNB"         : Pipeline([("bin", __import__("sklearn.preprocessing", fromlist=["Binarizer"]).Binarizer()),
                                       ("bnb", BernoulliNB())]),
}

print("\n  Dataset : Iris (150 x 4, 3 classes)")
names_all, accs_all, cvs_all = [], [], []
for name, clf in all_clfs.items():
    clf.fit(X_iris_tr, y_iris_tr)
    acc = accuracy_score(y_iris_te, clf.predict(X_iris_te))
    cv  = cross_val_score(clf, iris.data, iris.target, cv=5, scoring="accuracy")
    names_all.append(name); accs_all.append(acc); cvs_all.append(cv.mean())
    print(f"  {name:25s} | Test={acc:.4f} | CV={cv.mean():.4f}+/-{cv.std():.4f}")

fig, ax = plt.subplots(figsize=(12, 8))
sorted_idx  = np.argsort(accs_all)
sorted_names = [names_all[i] for i in sorted_idx]
sorted_accs  = [accs_all[i] for i in sorted_idx]
colors_all   = plt.cm.RdYlGn(np.array(sorted_accs))
bars_all = ax.barh(sorted_names, sorted_accs, color=colors_all)
ax.set_xlabel("Accuracy (test, Iris)")
ax.set_title("SYNTHESE GLOBALE -- Classifieurs supervisees (Iris)",
              fontsize=13, fontweight="bold")
ax.set_xlim(0.7, 1.05)
for bar, v in zip(bars_all, sorted_accs):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=7)
plt.tight_layout(); plt.show()

# --- Regression : comparaison recapitulative ---
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   BayesianRidge, HuberRegressor, SGDRegressor,
                                   RANSACRegressor, TheilSenRegressor,
                                   QuantileRegressor, ARDRegression)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

all_regs = {
    "LinearRegression"   : Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge(a=1)"         : Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso(a=0.1)"       : Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1, max_iter=5000))]),
    "ElasticNet"         : Pipeline([("sc", StandardScaler()), ("m", ElasticNet(max_iter=5000))]),
    "Bayesian Ridge"     : Pipeline([("sc", StandardScaler()), ("m", BayesianRidge())]),
    "Huber"              : Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(epsilon=1.35))]),
    "SGD Reg."           : Pipeline([("sc", StandardScaler()), ("m", SGDRegressor(penalty="l2", alpha=0.001, max_iter=1000, random_state=42))]),
    "Decision Tree"      : Pipeline([("sc", StandardScaler()), ("m", DecisionTreeRegressor(max_depth=6, random_state=42))]),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR (rbf)"          : Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=100))]),
    "KNN (k=7)"          : Pipeline([("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=7))]),
    "MLP Reg."           : Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))]),
    "Poly(deg=2)+Ridge"  : Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                      ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Quantile (Q50)"     : Pipeline([("sc", StandardScaler()), ("m", QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs"))]),
}

print("\n  Dataset : Diabetes (442 x 10)")
names_r, r2s_r = [], []
for name, reg in all_regs.items():
    reg.fit(X_dia_tr, y_dia_tr)
    pred = reg.predict(X_dia_te)
    if hasattr(pred, "ravel"): pred = pred.ravel()
    r2_v   = r2_score(y_dia_te, pred)
    rmse_v = np.sqrt(mean_squared_error(y_dia_te, pred))
    names_r.append(name); r2s_r.append(r2_v)
    print(f"  {name:22s} | R2={r2_v:.4f} | RMSE={rmse_v:.2f}")

fig, ax = plt.subplots(figsize=(12, 8))
s_idx   = np.argsort(r2s_r)
s_names = [names_r[i] for i in s_idx]
s_r2s   = [r2s_r[i] for i in s_idx]
colors_r = plt.cm.RdYlGn(np.array([max(r,0) for r in s_r2s]) /
                           max(max(s_r2s), 0.01))
bars_r = ax.barh(s_names, s_r2s, color=colors_r)
ax.set_xlabel("R2 Score (Diabetes, test set)")
ax.set_title("SYNTHESE GLOBALE -- Regresseurs supervises (Diabetes)",
              fontsize=13, fontweight="bold")
ax.axvline(0, color="black", lw=1)
for bar, v in zip(bars_r, s_r2s):
    ax.text(max(v, 0) + 0.005, bar.get_y() + bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=7)
plt.tight_layout(); plt.show()

print("\n[OK] Partie 3 terminee -- synthese globale generee avec succes.")
