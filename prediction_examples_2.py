"""
============================================================
ALGORITHMES DE PREDICTION SUPERVISEE -- Partie 2
============================================================
Algorithmes couverts :

  CLASSIFICATION (suite) :
    1.  Analyse Discriminante Lineaire (LDA)
    2.  Analyse Discriminante Quadratique (QDA)
    3.  SGD Classifier (Descente de Gradient Stochastique)
    4.  Bagging Classifier
    5.  Voting Classifier (Hard & Soft)
    6.  Stacking Classifier
    7.  Gaussian Process Classifier

  REGRESSION (suite) :
    8.  Bayesian Ridge Regression
    9.  Huber Regressor (robuste aux outliers)
    10. ARD Regression (Automatic Relevance Determination)
    11. Partial Least Squares (PLS Regression)
    12. SGD Regressor
    13. Bagging Regressor
    14. Stacking Regressor
    15. Gaussian Process Regressor

  OPTIMISATION DES HYPERPARAMETRES :
    16. GridSearchCV
    17. RandomizedSearchCV
    18. Cross-Validation avancee (StratifiedKFold, RepeatedKFold)
============================================================
"""

import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, RepeatedKFold,
                                      learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report,
                              mean_squared_error, r2_score,
                              mean_absolute_error)

# ============================================================
# DONNEES
# ============================================================
iris     = datasets.load_iris()
breast   = datasets.load_breast_cancer()
wine     = datasets.load_wine()
digits   = datasets.load_digits()
diabetes = datasets.load_diabetes()
california = datasets.fetch_california_housing()

# Splits classification
X_iris_tr, X_iris_te, y_iris_tr, y_iris_te = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(
    breast.data, breast.target, test_size=0.2, random_state=42, stratify=breast.target)
X_wine_tr, X_wine_te, y_wine_tr, y_wine_te = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)
X_dig_tr, X_dig_te, y_dig_tr, y_dig_te = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)

# Splits regression
X_dia_tr, X_dia_te, y_dia_tr, y_dia_te = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42)
X_cal_tr, X_cal_te, y_cal_tr, y_cal_te = train_test_split(
    california.data, california.target, test_size=0.2, random_state=42)

# Scaler reference
sc = StandardScaler()

# ============================================================
# UTILITAIRES
# ============================================================
def plot_cm(y_true, y_pred, names, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Prediction"); ax.set_ylabel("Reel")

def plot_scatter_reg(y_true, y_pred, title, ax):
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color="#3498db")
    mn = min(y_true.min(), y_pred.min()); mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    r2 = r2_score(y_true, y_pred)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Valeur reelle"); ax.set_ylabel("Valeur predite")
    ax.text(0.05, 0.92, f"R2={r2:.3f}", transform=ax.transAxes,
            fontsize=9, color="#e74c3c", fontweight="bold")

def reg_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:30s} | RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.4f}")
    return rmse, mae, r2

# ============================================================
# SECTION A : CLASSIFICATION (suite)
# ============================================================
print("=" * 65)
print("ALGORITHMES DE CLASSIFICATION -- Partie 2")
print("=" * 65)

clf_res = {}

# ============================================================
# 1. ANALYSE DISCRIMINANTE LINEAIRE (LDA)
# ============================================================
print("\n" + "=" * 65)
print("1. ANALYSE DISCRIMINANTE LINEAIRE (LDA)")
print("=" * 65)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA : classificateur ET reducteur de dimension (projette sur k-1 axes)
lda = LinearDiscriminantAnalysis()
lda.fit(X_iris_tr, y_iris_tr)
pred_lda = lda.predict(X_iris_te)
acc_lda  = accuracy_score(y_iris_te, pred_lda)
print(f"[Iris] Accuracy : {acc_lda:.4f}")
clf_res["LDA"] = acc_lda

# LDA comme reducteur de dimension (max k-1 = 2 composantes sur Iris 3 classes)
X_iris_lda    = lda.transform(iris.data)          # projection 4D -> 2D
X_wine_lda_tr = LinearDiscriminantAnalysis().fit(X_wine_tr, y_wine_tr).transform(X_wine_tr)
lda_wine      = LinearDiscriminantAnalysis().fit(X_wine_tr, y_wine_tr)
pred_lda_wine = lda_wine.predict(X_wine_te)
print(f"[Wine] Accuracy : {accuracy_score(y_wine_te, pred_lda_wine):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_iris_te, pred_lda, iris.target_names, "LDA -- Iris", axes[0])
# Projection LDA 2D
colors_lda = ["#e74c3c", "#2ecc71", "#3498db"]
for i, cls in enumerate(iris.target_names):
    mask = iris.target == i
    axes[1].scatter(X_iris_lda[mask, 0], X_iris_lda[mask, 1],
                    c=colors_lda[i], label=cls, s=30, alpha=0.8)
axes[1].set_title("LDA -- Projection 2D (Iris)", fontweight="bold")
axes[1].set_xlabel("LD1"); axes[1].set_ylabel("LD2"); axes[1].legend()
plt.suptitle("ANALYSE DISCRIMINANTE LINEAIRE (LDA)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 2. ANALYSE DISCRIMINANTE QUADRATIQUE (QDA)
# ============================================================
print("\n" + "=" * 65)
print("2. ANALYSE DISCRIMINANTE QUADRATIQUE (QDA)")
print("=" * 65)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda      = QuadraticDiscriminantAnalysis()
qda.fit(X_iris_tr, y_iris_tr)
pred_qda = qda.predict(X_iris_te)
acc_qda  = accuracy_score(y_iris_te, pred_qda)
proba_qda = qda.predict_proba(X_iris_te)
print(f"[Iris] Accuracy : {acc_qda:.4f}")
clf_res["QDA"] = acc_qda

# Comparison LDA vs QDA sur Wine
lda_w = LinearDiscriminantAnalysis().fit(X_wine_tr, y_wine_tr)
qda_w = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(X_wine_tr, y_wine_tr)
acc_lda_w = accuracy_score(y_wine_te, lda_w.predict(X_wine_te))
acc_qda_w = accuracy_score(y_wine_te, qda_w.predict(X_wine_te))
print(f"[Wine] LDA Accuracy : {acc_lda_w:.4f} | QDA Accuracy : {acc_qda_w:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_iris_te, pred_qda, iris.target_names, "QDA -- Iris", axes[0])
# Probabilites posterieures (QDA = sortie probabiliste)
x_range = np.arange(len(y_iris_te))
for i, cls in enumerate(iris.target_names):
    axes[1].plot(x_range, proba_qda[:, i], label=cls, lw=1.2)
axes[1].set_title("Probabilites QDA par classe -- Iris (test set)", fontweight="bold")
axes[1].set_xlabel("Indice echantillon"); axes[1].set_ylabel("P(classe|x)")
axes[1].legend(fontsize=9)
plt.suptitle("ANALYSE DISCRIMINANTE QUADRATIQUE (QDA)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 3. SGD CLASSIFIER
# ============================================================
print("\n" + "=" * 65)
print("3. SGD CLASSIFIER (Descente de Gradient Stochastique)")
print("=" * 65)

from sklearn.linear_model import SGDClassifier

# SGD avec differentes pertes
losses_sgd = {
    "hinge (SVM lineaire)" : "hinge",
    "log_loss (LR)"        : "log_loss",
    "modified_huber"       : "modified_huber",
}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name_loss, loss) in zip(axes, losses_sgd.items()):
    sgd = Pipeline([
        ("sc",  StandardScaler()),
        ("sgd", SGDClassifier(loss=loss, max_iter=1000,
                               random_state=42, n_jobs=-1))
    ])
    sgd.fit(X_dig_tr, y_dig_tr)
    pred_sgd = sgd.predict(X_dig_te)
    acc_sgd  = accuracy_score(y_dig_te, pred_sgd)
    print(f"[Digits] SGD loss={name_loss:25s} | Accuracy={acc_sgd:.4f}")
    if loss == "log_loss":
        clf_res["SGD (log)"] = acc_sgd
    plot_cm(y_dig_te, pred_sgd, [str(i) for i in range(10)],
            f"SGD ({name_loss}) -- Digits", ax)
plt.suptitle("SGD CLASSIFIER -- Comparaison des fonctions de perte (Digits)",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 4. BAGGING CLASSIFIER
# ============================================================
print("\n" + "=" * 65)
print("4. BAGGING CLASSIFIER")
print("=" * 65)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging de Decision Trees (= base de Random Forest)
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100,
    max_samples=0.8,       # 80% des echantillons par base learner
    max_features=0.8,      # 80% des features par base learner
    bootstrap=True,        # avec remise (bootstrap)
    random_state=42,
    n_jobs=-1
)
bag.fit(X_wine_tr, y_wine_tr)
pred_bag = bag.predict(X_wine_te)
acc_bag  = accuracy_score(y_wine_te, pred_bag)
print(f"[Wine] Bagging(DT x100) Accuracy : {acc_bag:.4f}")
clf_res["Bagging"] = acc_bag

# Comparaison : 1 arbre seul vs Bagging
dt_alone = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_alone.fit(X_wine_tr, y_wine_tr)
acc_dt = accuracy_score(y_wine_te, dt_alone.predict(X_wine_te))
print(f"[Wine] Arbre seul        Accuracy : {acc_dt:.4f}")

# Variance par bagging : OOB score
bag_oob = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=200, bootstrap=True, oob_score=True,
    random_state=42, n_jobs=-1
)
bag_oob.fit(X_wine_tr, y_wine_tr)
print(f"[Wine] OOB Score  : {bag_oob.oob_score_:.4f}")

# Evolution de l'accuracy avec nb d'estimateurs
n_range = [1, 5, 10, 20, 50, 100, 150, 200]
accs_bag = []
for n in n_range:
    b = BaggingClassifier(estimator=DecisionTreeClassifier(),
                           n_estimators=n, random_state=42, n_jobs=-1)
    b.fit(X_wine_tr, y_wine_tr)
    accs_bag.append(accuracy_score(y_wine_te, b.predict(X_wine_te)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_wine_te, pred_bag, wine.target_names, "Bagging -- Wine", axes[0])
axes[1].plot(n_range, accs_bag, "bo-", lw=2)
axes[1].axhline(acc_dt, color="red", linestyle="--", label=f"Arbre seul ({acc_dt:.3f})")
axes[1].set_xlabel("Nombre d'estimateurs"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Bagging : Accuracy vs nombre d'arbres", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("BAGGING CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 5. VOTING CLASSIFIER (Hard & Soft)
# ============================================================
print("\n" + "=" * 65)
print("5. VOTING CLASSIFIER (Hard & Soft)")
print("=" * 65)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Membres de l'ensemble
base_clfs = [
    ("lr",  Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))])),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("svm", Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=42))])),
]

# Hard Voting : vote majoritaire des classes predites
vote_hard = VotingClassifier(estimators=base_clfs, voting="hard")
vote_hard.fit(X_iris_tr, y_iris_tr)
pred_vh  = vote_hard.predict(X_iris_te)
acc_vh   = accuracy_score(y_iris_te, pred_vh)
print(f"[Iris] Hard Voting Accuracy : {acc_vh:.4f}")
clf_res["Hard Voting"] = acc_vh

# Soft Voting : moyenne des probabilites
vote_soft = VotingClassifier(estimators=base_clfs, voting="soft")
vote_soft.fit(X_iris_tr, y_iris_tr)
pred_vs  = vote_soft.predict(X_iris_te)
acc_vs   = accuracy_score(y_iris_te, pred_vs)
print(f"[Iris] Soft Voting Accuracy : {acc_vs:.4f}")
clf_res["Soft Voting"] = acc_vs

# Comparaison individuelle vs ensemble
indiv_accs = {}
for name, clf in base_clfs:
    clf.fit(X_iris_tr, y_iris_tr)
    indiv_accs[name] = accuracy_score(y_iris_te, clf.predict(X_iris_te))
    print(f"  {name:5s} seul  : {indiv_accs[name]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_iris_te, pred_vs, iris.target_names, "Soft Voting -- Iris", axes[0])
labels_v = list(indiv_accs.keys()) + ["Hard Voting", "Soft Voting"]
values_v = list(indiv_accs.values()) + [acc_vh, acc_vs]
colors_v = ["#95a5a6"]*4 + ["#e74c3c", "#e74c3c"]
bars_v   = axes[1].bar(labels_v, values_v, color=colors_v)
axes[1].set_title("Individuel vs Voting -- Iris", fontweight="bold")
axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0.9, 1.02)
axes[1].tick_params(axis="x", rotation=20)
for bar, v in zip(bars_v, values_v):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.002,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("VOTING CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 6. STACKING CLASSIFIER
# ============================================================
print("\n" + "=" * 65)
print("6. STACKING CLASSIFIER")
print("=" * 65)

from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Base learners (niveau 0)
base_estimators = [
    ("rf",  RandomForestClassifier(n_estimators=50, random_state=42)),
    ("gb",  GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ("knn", Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])),
    ("nb",  GaussianNB()),
]

# Meta-learner (niveau 1) : apprend sur les sorties des base learners
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,                     # validation croisee pour generer les predictions hors-echantillon
    stack_method="predict_proba",  # utilise les probabilites comme features pour le meta-learner
    passthrough=False         # False : le meta-learner ne voit que les sorties des base learners
)
stack.fit(X_bc_tr, y_bc_tr)
pred_stack = stack.predict(X_bc_te)
acc_stack  = accuracy_score(y_bc_te, pred_stack)
print(f"[Breast Cancer] Stacking Accuracy : {acc_stack:.4f}")
clf_res["Stacking"] = acc_stack

# Comparaison base learners vs stacking
print("  Comparaison avec les base learners :")
for name, est in base_estimators:
    est.fit(X_bc_tr, y_bc_tr)
    acc_bl = accuracy_score(y_bc_te, est.predict(X_bc_te))
    print(f"    {name:5s} : {acc_bl:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_cm(y_bc_te, pred_stack, breast.target_names, "Stacking -- Breast Cancer", axes[0])

# Architecture du stacking
ax2 = axes[1]; ax2.axis("off")
ax2.text(0.5, 0.95, "Architecture du Stacking", transform=ax2.transAxes,
          ha="center", va="top", fontsize=12, fontweight="bold")
arch_text = (
    "Niveau 0 (Base Learners)\n"
    "  - Random Forest (50 arbres)\n"
    "  - Gradient Boosting (50 arbres)\n"
    "  - KNN (k=5)\n"
    "  - Naive Bayes\n\n"
    "  [sorties : probabilites par classe]\n\n"
    "Niveau 1 (Meta-Learner)\n"
    "  - Logistic Regression\n\n"
    "  [apprend sur les sorties de niveau 0]"
)
ax2.text(0.05, 0.85, arch_text, transform=ax2.transAxes,
          va="top", fontsize=10, family="monospace",
          bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

plt.suptitle("STACKING CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 7. GAUSSIAN PROCESS CLASSIFIER
# ============================================================
print("\n" + "=" * 65)
print("7. GAUSSIAN PROCESS CLASSIFIER")
print("=" * 65)

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct

# Sur Iris (taille moderee : GP est O(n^3))
kernels_gpc = {
    "RBF (l=1.0)"    : RBF(length_scale=1.0),
    "Matern (nu=1.5)": Matern(length_scale=1.0, nu=1.5),
}

sc_iris = StandardScaler()
X_iris_tr_s = sc_iris.fit_transform(X_iris_tr)
X_iris_te_s = sc_iris.transform(X_iris_te)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (kname, kernel) in zip(axes, kernels_gpc.items()):
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gpc.fit(X_iris_tr_s, y_iris_tr)
    pred_gpc = gpc.predict(X_iris_te_s)
    acc_gpc  = accuracy_score(y_iris_te, pred_gpc)
    print(f"[Iris] GPC kernel={kname:20s} | Accuracy={acc_gpc:.4f}")
    clf_res[f"GPC ({kname[:3]})"] = acc_gpc
    plot_cm(y_iris_te, pred_gpc, iris.target_names,
            f"GPC ({kname}) -- Iris", ax)
plt.suptitle("GAUSSIAN PROCESS CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# COMPARAISON CLASSIFICATION PARTIE 2
# ============================================================
print("\n" + "=" * 65)
print("COMPARAISON GLOBALE -- CLASSIFICATION Partie 2 (Iris/BC/Wine)")
print("=" * 65)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import (BaggingClassifier, VotingClassifier,
                               StackingClassifier, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

all_clfs_iris = {
    "LDA"           : LinearDiscriminantAnalysis(),
    "QDA"           : QuadraticDiscriminantAnalysis(),
    "SGD (log_loss)": Pipeline([("sc", StandardScaler()), ("m", SGDClassifier(loss="log_loss", random_state=42))]),
    "Bagging"       : BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42, n_jobs=-1),
    "Hard Voting"   : VotingClassifier(estimators=[
                          ("lr", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])),
                          ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                          ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42))
                      ], voting="hard"),
    "Soft Voting"   : VotingClassifier(estimators=[
                          ("lr", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))])),
                          ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                          ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42))
                      ], voting="soft"),
    "Stacking"      : StackingClassifier(
                          estimators=[
                              ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                              ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
                          ],
                          final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                          cv=5),
    "GPC (RBF)"     : Pipeline([("sc", StandardScaler()), ("m", GaussianProcessClassifier(kernel=RBF(), random_state=42))]),
}

names_c2, accs_c2 = [], []
for name, clf in all_clfs_iris.items():
    clf.fit(X_iris_tr, y_iris_tr)
    acc = accuracy_score(y_iris_te, clf.predict(X_iris_te))
    cv  = cross_val_score(clf, iris.data, iris.target, cv=5, scoring="accuracy")
    names_c2.append(name); accs_c2.append(acc)
    print(f"{name:22s} | Accuracy={acc:.4f} | CV={cv.mean():.4f}+/-{cv.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
colors_c2 = plt.cm.tab10(np.linspace(0, 1, len(names_c2)))
bars = ax.barh(names_c2, accs_c2, color=colors_c2)
ax.set_xlabel("Accuracy (test, Iris)"); ax.set_xlim(0.85, 1.04)
ax.set_title("Comparaison Partie 2 -- Classification (Iris)", fontweight="bold")
for bar, acc in zip(bars, accs_c2):
    ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2,
             f"{acc:.4f}", va="center", fontsize=8)
plt.tight_layout(); plt.show()


# ============================================================
# SECTION B : REGRESSION (suite)
# ============================================================
print("\n" + "=" * 65)
print("ALGORITHMES DE REGRESSION -- Partie 2")
print("=" * 65)

reg_res = {}

# ============================================================
# 8. BAYESIAN RIDGE REGRESSION
# ============================================================
print("\n" + "=" * 65)
print("8. BAYESIAN RIDGE REGRESSION")
print("=" * 65)

from sklearn.linear_model import BayesianRidge

sc_dia = StandardScaler()
X_dia_tr_s = sc_dia.fit_transform(X_dia_tr)
X_dia_te_s = sc_dia.transform(X_dia_te)

bay_ridge = BayesianRidge(max_iter=500)
bay_ridge.fit(X_dia_tr_s, y_dia_tr)
pred_br, pred_br_std = bay_ridge.predict(X_dia_te_s, return_std=True)
# pred_br_std : ecart-type de l'incertitude sur chaque prediction

rmse, mae, r2 = reg_metrics("BayesianRidge", y_dia_te, pred_br)
reg_res["BayesianRidge"] = r2
print(f"  Incertitude moyenne : {pred_br_std.mean():.2f}")

# Intervalles de confiance
sorted_idx = np.argsort(y_dia_te)
y_sort     = y_dia_te[sorted_idx]
p_sort     = pred_br[sorted_idx]
s_sort     = pred_br_std[sorted_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_dia_te, pred_br, "Bayesian Ridge -- Diabetes", axes[0])
x_plot = np.arange(len(y_sort))
axes[1].plot(x_plot, y_sort,  "k-",  lw=1.5, label="Valeur reelle")
axes[1].plot(x_plot, p_sort,  "b--", lw=1.5, label="Prediction")
axes[1].fill_between(x_plot, p_sort - 2*s_sort, p_sort + 2*s_sort,
                      alpha=0.3, color="blue", label="Intervalle 95%")
axes[1].set_title("Predictions avec intervalles de confiance -- Diabetes",
                   fontweight="bold")
axes[1].set_xlabel("Echantillon (trie par y)"); axes[1].legend(fontsize=8)
plt.suptitle("BAYESIAN RIDGE REGRESSION", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 9. HUBER REGRESSOR (robuste aux outliers)
# ============================================================
print("\n" + "=" * 65)
print("9. HUBER REGRESSOR")
print("=" * 65)

from sklearn.linear_model import HuberRegressor, LinearRegression

# Ajout d'outliers artificiels pour demontrer la robustesse
rng = np.random.default_rng(42)
idx_outliers = rng.choice(len(X_dia_tr_s), size=20, replace=False)
y_dia_tr_noisy = y_dia_tr.copy()
y_dia_tr_noisy[idx_outliers] += 300   # outliers extremes

# Comparaison OLS vs Huber en presence d'outliers
ols_noisy   = LinearRegression().fit(X_dia_tr_s, y_dia_tr_noisy)
huber_r     = HuberRegressor(epsilon=1.35, max_iter=500)
huber_r.fit(X_dia_tr_s, y_dia_tr_noisy)

pred_ols_n   = ols_noisy.predict(X_dia_te_s)
pred_huber   = huber_r.predict(X_dia_te_s)

rmse_ols, _, r2_ols     = reg_metrics("OLS (avec outliers)", y_dia_te, pred_ols_n)
rmse_hub, _, r2_hub     = reg_metrics("Huber (avec outliers)", y_dia_te, pred_huber)
reg_res["Huber"] = r2_hub

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_dia_te, pred_ols_n,  f"OLS (R2={r2_ols:.3f}) -- avec outliers",  axes[0])
plot_scatter_reg(y_dia_te, pred_huber,   f"Huber (R2={r2_hub:.3f}) -- robuste",       axes[1])
plt.suptitle("HUBER REGRESSOR vs OLS en presence d'outliers -- Diabetes",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 10. ARD REGRESSION (Automatic Relevance Determination)
# ============================================================
print("\n" + "=" * 65)
print("10. ARD REGRESSION (Automatic Relevance Determination)")
print("=" * 65)

from sklearn.linear_model import ARDRegression, BayesianRidge

ard = ARDRegression(max_iter=500)
ard.fit(X_dia_tr_s, y_dia_tr)
pred_ard = ard.predict(X_dia_te_s)
rmse, mae, r2 = reg_metrics("ARD Regression", y_dia_te, pred_ard)
reg_res["ARD"] = r2

# ARD determine automatiquement la pertinence de chaque feature
# Lambda (precision des poids) : faible = feature pertinente
lambda_ = ard.lambda_   # un lambda par feature

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_dia_te, pred_ard, "ARD Regression -- Diabetes", axes[0])
feat_names_dia = [f"x{i}" for i in range(X_dia_tr_s.shape[1])]
# Plus lambda est faible, plus la feature est pertinente
relevance = 1.0 / lambda_
axes[1].barh(feat_names_dia, relevance, color="#9b59b6")
axes[1].set_title("Pertinence ARD par feature (1/lambda) -- Diabetes",
                   fontweight="bold")
axes[1].set_xlabel("Pertinence (1/lambda) -- plus grand = plus pertinent")
plt.suptitle("ARD REGRESSION", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 11. PARTIAL LEAST SQUARES (PLS Regression)
# ============================================================
print("\n" + "=" * 65)
print("11. PARTIAL LEAST SQUARES (PLS Regression)")
print("=" * 65)

from sklearn.cross_decomposition import PLSRegression

# PLS cherche les composantes qui maximisent la covariance X-Y
# Utile quand les features sont tres correlees

n_comp_range = range(1, X_dia_tr_s.shape[1]+1)
r2_pls = []
for n in n_comp_range:
    pls_cv = PLSRegression(n_components=n)
    pls_cv.fit(X_dia_tr_s, y_dia_tr)
    r2_pls.append(r2_score(y_dia_te, pls_cv.predict(X_dia_te_s)))

best_n = int(np.argmax(r2_pls)) + 1
pls    = PLSRegression(n_components=best_n)
pls.fit(X_dia_tr_s, y_dia_tr)
pred_pls = pls.predict(X_dia_te_s).ravel()
rmse, mae, r2 = reg_metrics(f"PLS (n_comp={best_n})", y_dia_te, pred_pls)
reg_res["PLS"] = r2
print(f"  Nombre de composantes optimal : {best_n}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_dia_te, pred_pls, f"PLS Regression (n={best_n}) -- Diabetes", axes[0])
axes[1].plot(list(n_comp_range), r2_pls, "bo-", lw=2)
axes[1].axvline(best_n, color="red", linestyle="--",
                 label=f"Optimal n={best_n}")
axes[1].set_xlabel("Nombre de composantes PLS"); axes[1].set_ylabel("R2 (test)")
axes[1].set_title("PLS : R2 vs nombre de composantes -- Diabetes", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("PARTIAL LEAST SQUARES (PLS)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 12. SGD REGRESSOR
# ============================================================
print("\n" + "=" * 65)
print("12. SGD REGRESSOR")
print("=" * 65)

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    loss="squared_error",   # MSE
    penalty="l2",           # regularisation Ridge
    alpha=0.001,
    max_iter=1000,
    tol=1e-4,
    random_state=42
)
sgd_reg.fit(X_dia_tr_s, y_dia_tr)
pred_sgdr = sgd_reg.predict(X_dia_te_s)
rmse, mae, r2 = reg_metrics("SGD Regressor", y_dia_te, pred_sgdr)
reg_res["SGD_R"] = r2

# Comparaison differentes penalites
penalties = {"l2 (Ridge)": "l2", "l1 (Lasso)": "l1", "elasticnet": "elasticnet"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (pname, pen) in zip(axes, penalties.items()):
    sgdr_p = SGDRegressor(loss="squared_error", penalty=pen,
                           alpha=0.001, max_iter=1000, random_state=42)
    sgdr_p.fit(X_dia_tr_s, y_dia_tr)
    pred_p = sgdr_p.predict(X_dia_te_s)
    r2_p   = r2_score(y_dia_te, pred_p)
    print(f"  SGD penalty={pname:15s} | R2={r2_p:.4f}")
    plot_scatter_reg(y_dia_te, pred_p, f"SGD Reg. ({pname}) -- Diabetes", ax)
plt.suptitle("SGD REGRESSOR -- Comparaison des penalites (Diabetes)",
             fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 13. BAGGING REGRESSOR
# ============================================================
print("\n" + "=" * 65)
print("13. BAGGING REGRESSOR")
print("=" * 65)

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=6),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
sc_cal = StandardScaler()
X_cal_tr_s = sc_cal.fit_transform(X_cal_tr)
X_cal_te_s = sc_cal.transform(X_cal_te)

bag_reg.fit(X_cal_tr_s, y_cal_tr)
pred_bag_r = bag_reg.predict(X_cal_te_s)
rmse, mae, r2 = reg_metrics("Bagging Regressor", y_cal_te, pred_bag_r)
reg_res["Bagging_R"] = r2

# Variance comparison : 1 arbre vs Bagging
dtr_alone = DecisionTreeRegressor(max_depth=6, random_state=42)
dtr_alone.fit(X_cal_tr_s, y_cal_tr)
pred_dtr_alone = dtr_alone.predict(X_cal_te_s)
r2_dtr = r2_score(y_cal_te, pred_dtr_alone)
rmse_dtr = np.sqrt(mean_squared_error(y_cal_te, pred_dtr_alone))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_cal_te, pred_bag_r, "Bagging Regressor -- California", axes[0])
axes[1].bar(["Arbre seul\n(max_depth=6)", "Bagging\n(100 arbres)"],
             [r2_dtr, r2],
             color=["#95a5a6", "#2ecc71"])
axes[1].set_ylabel("R2"); axes[1].set_ylim(0, 1.05)
axes[1].set_title("Arbre seul vs Bagging -- California", fontweight="bold")
for x, (label, val) in enumerate([("DT", r2_dtr), ("Bag", r2)]):
    axes[1].text(x, val + 0.01, f"{val:.4f}", ha="center", fontweight="bold")
plt.suptitle("BAGGING REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 14. STACKING REGRESSOR
# ============================================================
print("\n" + "=" * 65)
print("14. STACKING REGRESSOR")
print("=" * 65)

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

base_regs = [
    ("rf",  RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
    ("gb",  GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ("dtr", DecisionTreeRegressor(max_depth=6, random_state=42)),
]

stack_reg = StackingRegressor(
    estimators=base_regs,
    final_estimator=Ridge(alpha=1.0),   # meta-learner lineaire
    cv=5,
    n_jobs=-1
)
stack_reg.fit(X_cal_tr_s, y_cal_tr)
pred_sr  = stack_reg.predict(X_cal_te_s)
rmse, mae, r2 = reg_metrics("Stacking Regressor", y_cal_te, pred_sr)
reg_res["Stacking_R"] = r2

# Compare base learners vs stacking
print("  Comparaison base learners vs stacking :")
for name, reg in base_regs:
    reg.fit(X_cal_tr_s, y_cal_tr)
    r2_bl = r2_score(y_cal_te, reg.predict(X_cal_te_s))
    print(f"    {name} : R2={r2_bl:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_cal_te, pred_sr, "Stacking Regressor -- California", axes[0])
ax2 = axes[1]; ax2.axis("off")
ax2.text(0.5, 0.98, "Architecture Stacking Regressor", ha="center", va="top",
          fontsize=12, fontweight="bold", transform=ax2.transAxes)
arch_r = (
    "Niveau 0 (Base Regresseurs)\n"
    "  - Random Forest   (50 arbres)\n"
    "  - Gradient Boosting (50 arbres)\n"
    "  - Decision Tree   (depth=6)\n\n"
    "  [sorties : predictions numeriques]\n\n"
    "Niveau 1 (Meta-Learner)\n"
    "  - Ridge Regression (alpha=1.0)\n\n"
    "  [apprend a combiner les predictions]"
)
ax2.text(0.05, 0.85, arch_r, va="top", fontsize=10, family="monospace",
          transform=ax2.transAxes,
          bbox=dict(boxstyle="round", facecolor="#eaf4fb", alpha=0.9))
plt.suptitle("STACKING REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 15. GAUSSIAN PROCESS REGRESSOR
# ============================================================
print("\n" + "=" * 65)
print("15. GAUSSIAN PROCESS REGRESSOR")
print("=" * 65)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Sur Diabetes (taille raisonnable pour GP)
kernel_gpr = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel_gpr, n_restarts_optimizer=3,
                                 random_state=42)
gpr.fit(X_dia_tr_s, y_dia_tr)
pred_gpr, std_gpr = gpr.predict(X_dia_te_s, return_std=True)
rmse, mae, r2 = reg_metrics("Gaussian Process Regressor", y_dia_te, pred_gpr)
reg_res["GPR"] = r2
print(f"  Kernel optimise : {gpr.kernel_}")

# Incertitude : prediction + intervalle de confiance
sorted_idx = np.argsort(y_dia_te)
y_s = y_dia_te[sorted_idx]; p_s = pred_gpr[sorted_idx]; s_s = std_gpr[sorted_idx]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_dia_te, pred_gpr, "GPR -- Diabetes", axes[0])
x_p = np.arange(len(y_s))
axes[1].plot(x_p, y_s,  "k-",  lw=1.5, label="Valeur reelle")
axes[1].plot(x_p, p_s,  "b--", lw=1.5, label="GPR prediction")
axes[1].fill_between(x_p, p_s - 2*s_s, p_s + 2*s_s,
                      alpha=0.25, color="blue", label="95% CI")
axes[1].set_title("GPR avec incertitude -- Diabetes", fontweight="bold")
axes[1].set_xlabel("Echantillon (trie par y)"); axes[1].legend(fontsize=8)
plt.suptitle("GAUSSIAN PROCESS REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# SECTION C : OPTIMISATION DES HYPERPARAMETRES
# ============================================================
print("\n" + "=" * 65)
print("OPTIMISATION DES HYPERPARAMETRES")
print("=" * 65)


# ============================================================
# 16. GRIDSEARCHCV
# ============================================================
print("\n" + "=" * 65)
print("16. GRIDSEARCHCV -- Random Forest sur Iris")
print("=" * 65)

from sklearn.model_selection import GridSearchCV

rf_gs = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [None, 5, 10],
    "max_features" : ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}

gs = GridSearchCV(
    estimator=rf_gs,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
gs.fit(X_iris_tr, y_iris_tr)
print(f"  Meilleurs params : {gs.best_params_}")
print(f"  Meilleur score CV : {gs.best_score_:.4f}")
best_rf = gs.best_estimator_
pred_gs  = best_rf.predict(X_iris_te)
print(f"  Accuracy test    : {accuracy_score(y_iris_te, pred_gs):.4f}")

# Heatmap des resultats GridSearch (n_estimators x max_depth)
import pandas as pd
gs_results = pd.DataFrame(gs.cv_results_)
# Restriction a max_features=sqrt et min_samples_split=2
mask = (gs_results["param_max_features"] == "sqrt") & \
       (gs_results["param_min_samples_split"] == 2)
filtered = gs_results[mask].pivot_table(
    values="mean_test_score",
    index="param_max_depth",
    columns="param_n_estimators"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(filtered.values, cmap="YlOrRd", aspect="auto")
axes[0].set_xticks(range(len(filtered.columns)))
axes[0].set_yticks(range(len(filtered.index)))
axes[0].set_xticklabels(filtered.columns)
axes[0].set_yticklabels(filtered.index)
for i in range(len(filtered.index)):
    for j in range(len(filtered.columns)):
        axes[0].text(j, i, f"{filtered.values[i,j]:.3f}",
                      ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=axes[0])
axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("max_depth")
axes[0].set_title("GridSearchCV -- Accuracy CV (RF, Iris)", fontweight="bold")

plot_cm(y_iris_te, pred_gs, iris.target_names,
        f"RF apres GridSearchCV -- Iris", axes[1])
plt.suptitle("GRIDSEARCHCV", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 17. RANDOMIZEDSEARCHCV
# ============================================================
print("\n" + "=" * 65)
print("17. RANDOMIZEDSEARCHCV -- GBR sur Diabetes")
print("=" * 65)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

gbr_rs = GradientBoostingRegressor(random_state=42)
param_dist = {
    "n_estimators"  : randint(50, 400),
    "learning_rate" : uniform(0.01, 0.3),
    "max_depth"     : randint(2, 8),
    "subsample"     : uniform(0.6, 0.4),
    "min_samples_split" : randint(2, 10),
}

rs = RandomizedSearchCV(
    estimator=gbr_rs,
    param_distributions=param_dist,
    n_iter=40,                          # 40 combinaisons aleatoires
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=0
)
rs.fit(X_dia_tr_s, y_dia_tr)
print(f"  Meilleurs params : {rs.best_params_}")
print(f"  Meilleur score CV (R2) : {rs.best_score_:.4f}")
pred_rs = rs.best_estimator_.predict(X_dia_te_s)
rmse_rs, _, r2_rs = reg_metrics("GBR apres RandomSearch", y_dia_te, pred_rs)

# Courbe de convergence RandomSearch : score vs iteration
rs_df = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
top_scores = rs_df["mean_test_score"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_scatter_reg(y_dia_te, pred_rs,
                  f"GBR (RandomSearch) R2={r2_rs:.3f} -- Diabetes", axes[0])
axes[1].plot(np.arange(1, len(top_scores)+1), sorted(top_scores, reverse=True),
              "bo-", lw=1.5)
axes[1].set_xlabel("Rang des combinaisons testees"); axes[1].set_ylabel("Score R2 CV")
axes[1].set_title("RandomizedSearchCV -- Distribution des scores (GBR)", fontweight="bold")
axes[1].grid(True)
plt.suptitle("RANDOMIZEDSEARCHCV", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 18. CROSS-VALIDATION AVANCEE
# ============================================================
print("\n" + "=" * 65)
print("18. CROSS-VALIDATION AVANCEE")
print("=" * 65)

from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                      RepeatedKFold, LeaveOneOut,
                                      cross_validate)

rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

cv_strategies = {
    "StratifiedKFold (5)"         : StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    "StratifiedKFold (10)"        : StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    "Repeated Stratified (5x3)"   : RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
}

print("  Comparaison des strategies de CV (Random Forest sur Breast Cancer) :")
cv_means, cv_stds, cv_names = [], [], []
for name, cv_strat in cv_strategies.items():
    scores = cross_validate(rf_cv, breast.data, breast.target,
                             cv=cv_strat, scoring=["accuracy"],
                             n_jobs=-1)
    m = scores["test_accuracy"].mean()
    s = scores["test_accuracy"].std()
    cv_means.append(m); cv_stds.append(s); cv_names.append(name)
    print(f"  {name:35s} | {m:.4f} +/- {s:.4f}")

# Comparaison holdout vs CV
acc_holdout = accuracy_score(y_bc_te, RandomForestClassifier(n_estimators=100, random_state=42)
                              .fit(X_bc_tr, y_bc_tr).predict(X_bc_te))
print(f"  Holdout simple (80/20)             : {acc_holdout:.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
bars_cv = ax.barh(cv_names, cv_means, xerr=cv_stds, capsize=4,
                   color=plt.cm.tab10(np.linspace(0, 1, len(cv_names))), alpha=0.8)
ax.axvline(acc_holdout, color="red", linestyle="--",
            label=f"Holdout simple ({acc_holdout:.4f})")
ax.set_xlabel("Accuracy"); ax.set_xlim(0.92, 1.01)
ax.set_title("Comparaison des strategies de Cross-Validation -- RF sur Breast Cancer",
              fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="x")
plt.tight_layout(); plt.show()


# ============================================================
# TABLEAU DE SYNTHESE FINAL
# ============================================================
print("\n" + "=" * 65)
print("SYNTHESE GLOBALE -- REGRESSION Partie 2 (Diabetes)")
print("=" * 65)

all_regs_dia = {
    "Bayesian Ridge"     : BayesianRidge(max_iter=500),
    "Huber"              : HuberRegressor(epsilon=1.35),
    "ARD Regression"     : ARDRegression(max_iter=300),
    "PLS"                : PLSRegression(n_components=5),
    "SGD (l2)"           : SGDRegressor(penalty="l2", alpha=0.001, max_iter=1000, random_state=42),
    "Bagging (DT)"       : BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=6),
                                             n_estimators=100, random_state=42, n_jobs=-1),
    "Stacking"           : StackingRegressor(
                               estimators=[
                                   ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
                                   ("gb", GradientBoostingRegressor(n_estimators=50, random_state=42)),
                               ],
                               final_estimator=Ridge(alpha=1.0), cv=5),
    "GPR"                : GaussianProcessRegressor(kernel=C(1.0)*RBF(), random_state=42),
}

names_r2, r2s_final = [], []
for name, reg in all_regs_dia.items():
    reg.fit(X_dia_tr_s, y_dia_tr)
    prd = reg.predict(X_dia_te_s)
    if hasattr(prd, "ravel"):
        prd = prd.ravel()
    r2_f  = r2_score(y_dia_te, prd)
    rmse_f = np.sqrt(mean_squared_error(y_dia_te, prd))
    names_r2.append(name); r2s_final.append(r2_f)
    print(f"{name:22s} | R2={r2_f:.4f} | RMSE={rmse_f:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
colors_r2 = plt.cm.RdYlGn(np.array(r2s_final) / max(r2s_final))
bars_r2 = ax.barh(names_r2, r2s_final, color=colors_r2)
ax.set_xlabel("R2 (test, Diabetes)"); ax.set_xlim(0, 1.0)
ax.set_title("Synthese Partie 2 -- Regression (Diabetes)", fontweight="bold")
for bar, r2 in zip(bars_r2, r2s_final):
    ax.text(r2 + 0.005, bar.get_y() + bar.get_height()/2,
             f"{r2:.4f}", va="center", fontsize=8)
plt.tight_layout(); plt.show()

print("\n[OK] Script Partie 2 termine avec succes.")
