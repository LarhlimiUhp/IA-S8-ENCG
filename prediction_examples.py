"""
============================================================
ALGORITHMES DE PREDICTION SUPERVISEE -- Exemples avec donnees reelles
============================================================
Algorithmes couverts :

  CLASSIFICATION :
    1.  Regression Logistique
    2.  Decision Tree Classifier
    3.  Random Forest Classifier
    4.  Gradient Boosting Classifier
    5.  Support Vector Machine (SVM)
    6.  K-Nearest Neighbors (KNN)
    7.  Naive Bayes (GaussianNB)
    8.  AdaBoost Classifier
    9.  Perceptron Multicouche (MLP)
    10. Extra Trees Classifier

  REGRESSION :
    11. Regression Lineaire (OLS)
    12. Ridge / Lasso / ElasticNet
    13. Decision Tree Regressor
    14. Random Forest Regressor
    15. Gradient Boosting Regressor
    16. Support Vector Regression (SVR)
    17. KNN Regressor
    18. MLP Regressor

  COMPARAISONS GLOBALES :
    - Classification : Iris, Breast Cancer, Digits
    - Regression : California Housing, Diabetes

Installation des librairies optionnelles :
    pip install xgboost
============================================================
"""

import sys
import io
# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline

# ============================================================
# CHARGEMENT ET PREPARATION DES DONNEES
# ============================================================

# --- Datasets de classification ---
iris   = datasets.load_iris()
breast = datasets.load_breast_cancer()
wine   = datasets.load_wine()
digits = datasets.load_digits()

# --- Datasets de regression ---
california = datasets.fetch_california_housing()
diabetes   = datasets.load_diabetes()

# --- Splits train/test (80/20, stratified pour classification) ---
X_iris_tr, X_iris_te, y_iris_tr, y_iris_te = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = train_test_split(
    breast.data, breast.target, test_size=0.2, random_state=42, stratify=breast.target)

X_wine_tr, X_wine_te, y_wine_tr, y_wine_te = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)

X_dig_tr, X_dig_te, y_dig_tr, y_dig_te = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)

X_cal_tr, X_cal_te, y_cal_tr, y_cal_te = train_test_split(
    california.data, california.target, test_size=0.2, random_state=42)

X_dia_tr, X_dia_te, y_dia_tr, y_dia_te = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# --- Utilitaire : rapport rapide ---
def rapport_classification(name, y_true, y_pred, dataset_name, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    if target_names is not None:
        print(classification_report(y_true, y_pred, target_names=target_names,
                                    zero_division=0))
    return acc

def rapport_regression(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  RMSE : {rmse:.4f} | MAE : {mae:.4f} | R2 : {r2:.4f}")
    return rmse, mae, r2

def plot_confusion_matrix(y_true, y_pred, target_names, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(target_names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Prediction"); ax.set_ylabel("Reel")

def plot_reg_scatter(y_true, y_pred, title, ax):
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#3498db")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="y=x")
    ax.set_xlabel("Valeur reelle"); ax.set_ylabel("Valeur predite")
    ax.set_title(title, fontsize=10, fontweight="bold")
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"R2 = {r2:.3f}", transform=ax.transAxes, fontsize=9,
            color="#e74c3c", fontweight="bold")

# ============================================================
# SECTION A : ALGORITHMES DE CLASSIFICATION
# ============================================================

print("=" * 60)
print("ALGORITHMES DE CLASSIFICATION")
print("=" * 60)

clf_results = {}  # stocke les resultats pour comparaison finale

# ============================================================
# 1. REGRESSION LOGISTIQUE -- Dataset : Iris + Breast Cancer
# ============================================================
print("\n" + "=" * 60)
print("1. REGRESSION LOGISTIQUE")
print("=" * 60)

from sklearn.linear_model import LogisticRegression

# Pipeline : normalisation + modele
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(max_iter=1000, random_state=42))
])

# --- Iris (multi-classe) ---
lr_pipe.fit(X_iris_tr, y_iris_tr)
pred_lr_iris = lr_pipe.predict(X_iris_te)
print("[Iris - 3 classes]")
acc = rapport_classification("LR", y_iris_te, pred_lr_iris, "Iris",
                              target_names=iris.target_names)
clf_results["LogReg"] = acc

# --- Breast Cancer (binaire) ---
lr_bc = Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=1000, random_state=42))])
lr_bc.fit(X_bc_tr, y_bc_tr)
pred_lr_bc = lr_bc.predict(X_bc_te)
print("[Breast Cancer - binaire]")
rapport_classification("LR", y_bc_te, pred_lr_bc, "BreastCancer",
                        target_names=breast.target_names)

# --- Coefficients (modele linéaire interpretable) ---
coefs = lr_pipe["lr"].coef_  # (n_classes, n_features)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_iris_te, pred_lr_iris, iris.target_names,
                      "Reg. Logistique -- Iris", axes[0])
axes[1].bar(iris.feature_names, np.abs(coefs).mean(axis=0), color="#3498db")
axes[1].set_title("Importance des features (|coef| moyen)", fontweight="bold")
axes[1].set_xlabel("Feature"); axes[1].set_ylabel("|Coefficient|")
axes[1].tick_params(axis="x", rotation=20)
plt.suptitle("REGRESSION LOGISTIQUE", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 2. DECISION TREE CLASSIFIER -- Dataset : Wine
# ============================================================
print("\n" + "=" * 60)
print("2. DECISION TREE CLASSIFIER")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
dt.fit(X_wine_tr, y_wine_tr)
pred_dt = dt.predict(X_wine_te)
print("[Wine - 3 classes]")
acc = rapport_classification("DT", y_wine_te, pred_dt, "Wine",
                              target_names=wine.target_names)
clf_results["DecisionTree"] = acc

# Feature importances
importances = dt.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_confusion_matrix(y_wine_te, pred_dt, wine.target_names,
                      "Decision Tree -- Wine", axes[0])
axes[1].barh([wine.feature_names[i] for i in sorted_idx],
              importances[sorted_idx], color="#2ecc71")
axes[1].set_title("Feature Importances -- Decision Tree", fontweight="bold")
axes[1].set_xlabel("Importance (Gini)")
plt.suptitle("DECISION TREE CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# --- Visualisation de l'arbre ---
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True,
          rounded=True, fontsize=7, ax=ax)
plt.suptitle("Arbre de Decision -- Wine (max_depth=4)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 3. RANDOM FOREST CLASSIFIER -- Dataset : Iris + Breast Cancer
# ============================================================
print("\n" + "=" * 60)
print("3. RANDOM FOREST CLASSIFIER")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=None,
                             min_samples_split=2, random_state=42)
rf.fit(X_iris_tr, y_iris_tr)
pred_rf = rf.predict(X_iris_te)
print("[Iris - 3 classes]")
acc = rapport_classification("RF", y_iris_te, pred_rf, "Iris",
                              target_names=iris.target_names)
clf_results["RandomForest"] = acc

# --- Breast Cancer ---
rf_bc = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bc.fit(X_bc_tr, y_bc_tr)
pred_rf_bc = rf_bc.predict(X_bc_te)
print("[Breast Cancer - binaire]")
rapport_classification("RF", y_bc_te, pred_rf_bc, "BC",
                        target_names=breast.target_names)

# --- Learning Curve ---
scaler_rf = StandardScaler()
X_iris_scaled = scaler_rf.fit_transform(iris.data)
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_iris_scaled, iris.target,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_iris_te, pred_rf, iris.target_names,
                      "Random Forest -- Iris", axes[0])
axes[1].plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2ecc71",
              label="Train")
axes[1].fill_between(train_sizes,
                      train_scores.mean(1) - train_scores.std(1),
                      train_scores.mean(1) + train_scores.std(1), alpha=0.15,
                      color="#2ecc71")
axes[1].plot(train_sizes, test_scores.mean(axis=1), "o-", color="#e74c3c",
              label="Validation")
axes[1].fill_between(train_sizes,
                      test_scores.mean(1) - test_scores.std(1),
                      test_scores.mean(1) + test_scores.std(1), alpha=0.15,
                      color="#e74c3c")
axes[1].set_xlabel("Taille d'entrainement"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Learning Curve -- Random Forest", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("RANDOM FOREST CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 4. GRADIENT BOOSTING CLASSIFIER -- Dataset : Breast Cancer
# ============================================================
print("\n" + "=" * 60)
print("4. GRADIENT BOOSTING CLASSIFIER")
print("=" * 60)

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                  max_depth=3, random_state=42)
gb.fit(X_bc_tr, y_bc_tr)
pred_gb = gb.predict(X_bc_te)
print("[Breast Cancer - binaire]")
acc = rapport_classification("GB", y_bc_te, pred_gb, "BC",
                              target_names=breast.target_names)
clf_results["GradientBoosting"] = acc

# Probabilites de prediction
proba_gb = gb.predict_proba(X_bc_te)[:, 1]  # probabilite classe maligne

# --- Deviance (perte d'entrainement par iteration) ---
train_score = gb.train_score_

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_bc_te, pred_gb, breast.target_names,
                      "Gradient Boosting -- Breast Cancer", axes[0])
axes[1].plot(np.arange(gb.n_estimators) + 1, train_score, "b-", lw=1.5,
              label="Train Deviance")
axes[1].set_xlabel("Iterations (nb d'arbres)"); axes[1].set_ylabel("Deviance")
axes[1].set_title("Convergence Gradient Boosting", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("GRADIENT BOOSTING CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 5. SUPPORT VECTOR MACHINE (SVM) -- Dataset : Iris + Digits
# ============================================================
print("\n" + "=" * 60)
print("5. SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

from sklearn.svm import SVC

# Comparaison des kernels sur Iris
kernels = ["linear", "poly", "rbf", "sigmoid"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, kernel in zip(axes, kernels):
    svm_pipe = Pipeline([("sc", StandardScaler()),
                          ("svm", SVC(kernel=kernel, C=1.0, random_state=42))])
    svm_pipe.fit(X_iris_tr, y_iris_tr)
    pred_svm = svm_pipe.predict(X_iris_te)
    acc_svm  = accuracy_score(y_iris_te, pred_svm)
    print(f"[Iris] Kernel = {kernel:8s} | Accuracy : {acc_svm:.4f}")
    if kernel == "rbf":
        clf_results["SVM_RBF"] = acc_svm
    plot_confusion_matrix(y_iris_te, pred_svm, iris.target_names,
                          f"SVM ({kernel}) -- Iris", ax)

plt.suptitle("SVM -- Comparaison des Kernels (Iris)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# SVM sur Digits (haute dimension)
svm_digits = Pipeline([("sc", StandardScaler()),
                        ("svm", SVC(kernel="rbf", C=5.0, gamma="scale", random_state=42))])
svm_digits.fit(X_dig_tr, y_dig_tr)
pred_svm_dig = svm_digits.predict(X_dig_te)
print(f"\n[Digits - 10 classes] SVM RBF | Accuracy : {accuracy_score(y_dig_te, pred_svm_dig):.4f}")


# ============================================================
# 6. K-NEAREST NEIGHBORS (KNN) -- Dataset : Iris + Wine
# ============================================================
print("\n" + "=" * 60)
print("6. K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

from sklearn.neighbors import KNeighborsClassifier

# Effet du nombre de voisins k
k_range = range(1, 21)
acc_k   = []
for k in k_range:
    knn = Pipeline([("sc", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=k))])
    knn.fit(X_iris_tr, y_iris_tr)
    acc_k.append(accuracy_score(y_iris_te, knn.predict(X_iris_te)))

knn_best = Pipeline([("sc", StandardScaler()),
                      ("knn", KNeighborsClassifier(n_neighbors=5))])
knn_best.fit(X_iris_tr, y_iris_tr)
pred_knn = knn_best.predict(X_iris_te)
print(f"[Iris] KNN (k=5) | Accuracy : {accuracy_score(y_iris_te, pred_knn):.4f}")
clf_results["KNN"] = accuracy_score(y_iris_te, pred_knn)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_iris_te, pred_knn, iris.target_names,
                      "KNN (k=5) -- Iris", axes[0])
axes[1].plot(list(k_range), acc_k, "bo-", lw=2)
axes[1].axvline(x=5, color="red", linestyle="--", label="k=5 selectionne")
axes[1].set_xlabel("Nombre de voisins k"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy vs k -- KNN (Iris)", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("K-NEAREST NEIGHBORS", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 7. NAIVE BAYES (GaussianNB) -- Dataset : Iris + Breast Cancer
# ============================================================
print("\n" + "=" * 60)
print("7. NAIVE BAYES (GaussianNB)")
print("=" * 60)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_iris_tr, y_iris_tr)  # Naive Bayes n'a pas besoin de normalisation
pred_gnb = gnb.predict(X_iris_te)
proba_gnb = gnb.predict_proba(X_iris_te)
print(f"[Iris] Accuracy : {accuracy_score(y_iris_te, pred_gnb):.4f}")
clf_results["NaiveBayes"] = accuracy_score(y_iris_te, pred_gnb)

gnb_bc = GaussianNB()
gnb_bc.fit(X_bc_tr, y_bc_tr)
pred_gnb_bc = gnb_bc.predict(X_bc_te)
print(f"[Breast Cancer] Accuracy : {accuracy_score(y_bc_te, pred_gnb_bc):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_iris_te, pred_gnb, iris.target_names,
                      "Naive Bayes -- Iris", axes[0])
# Probabilites posterieures sur les 30 premiers echantillons de test
x = np.arange(30)
width = 0.25
for i, cls in enumerate(iris.target_names):
    axes[1].bar(x + i*width, proba_gnb[:30, i], width, label=cls, alpha=0.8)
axes[1].set_xlabel("Echantillon de test")
axes[1].set_ylabel("Probabilite posterieure P(classe|x)")
axes[1].set_title("Probabilites Naive Bayes -- Iris (30 premiers)", fontweight="bold")
axes[1].legend(fontsize=8)
plt.suptitle("NAIVE BAYES (GaussianNB)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 8. ADABOOST CLASSIFIER -- Dataset : Breast Cancer
# ============================================================
print("\n" + "=" * 60)
print("8. ADABOOST CLASSIFIER")
print("=" * 60)

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                          algorithm="SAMME", random_state=42)
ada.fit(X_bc_tr, y_bc_tr)
pred_ada = ada.predict(X_bc_te)
print(f"[Breast Cancer] Accuracy : {accuracy_score(y_bc_te, pred_ada):.4f}")
clf_results["AdaBoost"] = accuracy_score(y_bc_te, pred_ada)

# Evolution de l'accuracy par iteration
staged_acc = [accuracy_score(y_bc_te, pred)
              for pred in ada.staged_predict(X_bc_te)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_bc_te, pred_ada, breast.target_names,
                      "AdaBoost -- Breast Cancer", axes[0])
axes[1].plot(np.arange(1, len(staged_acc)+1), staged_acc, "g-", lw=1.5)
axes[1].set_xlabel("Nombre d'estimateurs"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Convergence AdaBoost -- Breast Cancer", fontweight="bold")
axes[1].grid(True)
plt.suptitle("ADABOOST CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 9. PERCEPTRON MULTICOUCHE MLP -- Dataset : Digits
# ============================================================
print("\n" + "=" * 60)
print("9. PERCEPTRON MULTICOUCHE (MLP Neural Network)")
print("=" * 60)

from sklearn.neural_network import MLPClassifier

mlp = Pipeline([
    ("sc",  StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3 couches cachees
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    ))
])
mlp.fit(X_dig_tr, y_dig_tr)
pred_mlp = mlp.predict(X_dig_te)
print(f"[Digits - 10 classes] MLP | Accuracy : {accuracy_score(y_dig_te, pred_mlp):.4f}")
clf_results["MLP"] = accuracy_score(y_dig_te, pred_mlp)

# Courbe de perte
loss_curve = mlp["mlp"].loss_curve_
val_curve  = mlp["mlp"].validation_scores_

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_dig_te, pred_mlp, [str(i) for i in range(10)],
                      "MLP -- Digits", axes[0])
axes[1].plot(loss_curve, "b-", lw=1.5, label="Train Loss")
axes[1].plot(val_curve,  "r-", lw=1.5, label="Validation Accuracy")
axes[1].set_xlabel("Itérations"); axes[1].set_ylabel("Valeur")
axes[1].set_title("Courbes d'entrainement MLP -- Digits", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("PERCEPTRON MULTICOUCHE (MLP)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# Visualisation des poids de la premiere couche cachee
weights_l1 = mlp["mlp"].coefs_[0]  # (64 features, 256 neurones)
fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for ax, w in zip(axes.flatten(), weights_l1.T[:32]):
    ax.imshow(w.reshape(8, 8), cmap="seismic", vmin=-1, vmax=1)
    ax.axis("off")
plt.suptitle("Poids de la 1ere couche cachee MLP (32/256 neurones) -- Digits",
             fontsize=11, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 10. EXTRA TREES CLASSIFIER -- Dataset : Wine + Iris
# ============================================================
print("\n" + "=" * 60)
print("10. EXTRA TREES CLASSIFIER")
print("=" * 60)

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=200, random_state=42)
et.fit(X_wine_tr, y_wine_tr)
pred_et = et.predict(X_wine_te)
print(f"[Wine] Accuracy : {accuracy_score(y_wine_te, pred_et):.4f}")
clf_results["ExtraTrees"] = accuracy_score(y_wine_te, pred_et)

# Feature importances
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_confusion_matrix(y_wine_te, pred_et, wine.target_names,
                      "Extra Trees -- Wine", axes[0])
importances_et = et.feature_importances_
idx_et = np.argsort(importances_et)[::-1]
axes[1].barh([wine.feature_names[i] for i in idx_et], importances_et[idx_et],
              color="#9b59b6")
axes[1].set_title("Feature Importances -- Extra Trees (Wine)", fontweight="bold")
plt.suptitle("EXTRA TREES CLASSIFIER", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# COMPARAISON GLOBALE CLASSIFICATION
# ============================================================
print("\n" + "=" * 60)
print("COMPARAISON GLOBALE -- CLASSIFICATION (Iris + Breast Cancer)")
print("=" * 60)

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier, GradientBoostingClassifier,
                                    AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.naive_bayes   import GaussianNB
from sklearn.neural_network import MLPClassifier

classifiers_iris = {
    "LogReg"           : Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
    "DecisionTree"     : DecisionTreeClassifier(max_depth=4, random_state=42),
    "RandomForest"     : RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting" : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM RBF"          : Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", random_state=42))]),
    "KNN (k=5)"        : Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
    "NaiveBayes"       : GaussianNB(),
    "AdaBoost"         : AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42),
    "MLP"              : Pipeline([("sc", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))]),
    "ExtraTrees"       : ExtraTreesClassifier(n_estimators=100, random_state=42),
}

names, accs, cv_means, cv_stds = [], [], [], []
for name, clf in classifiers_iris.items():
    clf.fit(X_iris_tr, y_iris_tr)
    acc = accuracy_score(y_iris_te, clf.predict(X_iris_te))
    cv  = cross_val_score(clf, iris.data, iris.target, cv=5, scoring="accuracy")
    names.append(name); accs.append(acc)
    cv_means.append(cv.mean()); cv_stds.append(cv.std())
    print(f"{name:22s} | Test Acc : {acc:.4f} | CV : {cv.mean():.4f} +/- {cv.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
bars = axes[0].barh(names, accs, color=colors)
axes[0].set_xlabel("Accuracy (test set)")
axes[0].set_title("Accuracy Test -- Tous les classifieurs (Iris)", fontweight="bold")
axes[0].set_xlim(0.8, 1.02)
for bar, acc in zip(bars, accs):
    axes[0].text(acc + 0.002, bar.get_y() + bar.get_height()/2,
                  f"{acc:.4f}", va="center", fontsize=8)

axes[1].barh(names, cv_means, xerr=cv_stds, color=colors, alpha=0.8, capsize=4)
axes[1].set_xlabel("Accuracy CV 5-fold")
axes[1].set_title("Cross-Validation (5-fold) -- Tous les classifieurs (Iris)", fontweight="bold")
axes[1].set_xlim(0.8, 1.05)
plt.suptitle("COMPARAISON GLOBALE -- CLASSIFICATION (Iris)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# SECTION B : ALGORITHMES DE REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("ALGORITHMES DE REGRESSION")
print("=" * 60)

reg_results = {}  # stocke les R2 pour comparaison finale

# ============================================================
# 11. REGRESSION LINEAIRE (OLS) -- Dataset : California Housing
# ============================================================
print("\n" + "=" * 60)
print("11. REGRESSION LINEAIRE (OLS)")
print("=" * 60)

from sklearn.linear_model import LinearRegression

lr_reg = Pipeline([("sc", StandardScaler()),
                    ("lr", LinearRegression())])
lr_reg.fit(X_cal_tr, y_cal_tr)
pred_lr_reg = lr_reg.predict(X_cal_te)

print("[California Housing]")
rmse, mae, r2 = rapport_regression("LinearReg", y_cal_te, pred_lr_reg)
reg_results["LinearReg"] = r2

# Residus
residuals = y_cal_te - pred_lr_reg
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_reg_scatter(y_cal_te, pred_lr_reg, "Reg. Lineaire -- California Housing", axes[0])
axes[1].hist(residuals, bins=50, color="#3498db", edgecolor="white")
axes[1].axvline(0, color="red", lw=2, linestyle="--")
axes[1].set_title("Distribution des residus", fontweight="bold")
axes[1].set_xlabel("Residu (y - y_pred)")
axes[2].scatter(pred_lr_reg, residuals, alpha=0.3, s=10, color="#2ecc71")
axes[2].axhline(0, color="red", lw=2, linestyle="--")
axes[2].set_xlabel("y_pred"); axes[2].set_ylabel("Residu")
axes[2].set_title("Residus vs Predictions", fontweight="bold")
plt.suptitle("REGRESSION LINEAIRE (OLS)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 12. RIDGE / LASSO / ELASTICNET -- Dataset : Diabetes
# ============================================================
print("\n" + "=" * 60)
print("12. RIDGE / LASSO / ELASTICNET")
print("=" * 60)

from sklearn.linear_model import Ridge, Lasso, ElasticNet

regularized_regs = {
    "Ridge"      : Ridge(alpha=1.0),
    "Lasso"      : Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet" : ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
}

scaler_dia = StandardScaler()
X_dia_tr_s = scaler_dia.fit_transform(X_dia_tr)
X_dia_te_s = scaler_dia.transform(X_dia_te)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, regularized_regs.items()):
    model.fit(X_dia_tr_s, y_dia_tr)
    pred = model.predict(X_dia_te_s)
    rmse, mae, r2 = rapport_regression(name, y_dia_te, pred)
    print(f"  [{name}] RMSE={rmse:.2f} | R2={r2:.4f}")
    reg_results[name] = r2
    plot_reg_scatter(y_dia_te, pred, f"{name} -- Diabetes", ax)

plt.suptitle("RIDGE / LASSO / ELASTICNET -- Diabetes", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# Effet de alpha sur les coefficients (Lasso)
alphas = np.logspace(-3, 2, 50)
coef_paths = []
feat_names = [f"x{i}" for i in range(X_dia_tr_s.shape[1])]
for a in alphas:
    m = Lasso(alpha=a, max_iter=10000).fit(X_dia_tr_s, y_dia_tr)
    coef_paths.append(m.coef_)
coef_paths = np.array(coef_paths)

fig, ax = plt.subplots(figsize=(10, 5))
for j in range(coef_paths.shape[1]):
    ax.plot(np.log10(alphas), coef_paths[:, j], label=feat_names[j])
ax.set_xlabel("log10(alpha)"); ax.set_ylabel("Coefficient")
ax.set_title("Chemin de regularisation Lasso -- Diabetes", fontweight="bold")
ax.legend(loc="upper right", fontsize=7, ncol=2)
ax.axvline(np.log10(0.1), color="red", linestyle="--", label="alpha=0.1 selectionne")
ax.grid(True)
plt.tight_layout(); plt.show()


# ============================================================
# 13. DECISION TREE REGRESSOR -- Dataset : California Housing
# ============================================================
print("\n" + "=" * 60)
print("13. DECISION TREE REGRESSOR")
print("=" * 60)

from sklearn.tree import DecisionTreeRegressor

# Effet de la profondeur
depths = [2, 4, 6, 8, None]
train_r2s, test_r2s = [], []
for d in depths:
    dtr = Pipeline([("sc", StandardScaler()),
                    ("dt", DecisionTreeRegressor(max_depth=d, random_state=42))])
    dtr.fit(X_cal_tr, y_cal_tr)
    train_r2s.append(r2_score(y_cal_tr, dtr.predict(X_cal_tr)))
    test_r2s.append(r2_score(y_cal_te, dtr.predict(X_cal_te)))

# Modele optimal max_depth=6
dtr_best = Pipeline([("sc", StandardScaler()),
                      ("dt", DecisionTreeRegressor(max_depth=6, random_state=42))])
dtr_best.fit(X_cal_tr, y_cal_tr)
pred_dtr = dtr_best.predict(X_cal_te)
rmse, mae, r2 = rapport_regression("DT", y_cal_te, pred_dtr)
reg_results["DecisionTree_R"] = r2
print(f"  [California] max_depth=6 | R2={r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg_scatter(y_cal_te, pred_dtr, "Decision Tree Regressor -- California", axes[0])
depth_labels = [str(d) if d else "None" for d in depths]
axes[1].plot(depth_labels, train_r2s, "bo-", label="Train R2")
axes[1].plot(depth_labels, test_r2s,  "ro-", label="Test R2")
axes[1].set_xlabel("max_depth"); axes[1].set_ylabel("R2")
axes[1].set_title("Effet de max_depth -- Decision Tree Regressor", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("DECISION TREE REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 14. RANDOM FOREST REGRESSOR -- Dataset : California Housing
# ============================================================
print("\n" + "=" * 60)
print("14. RANDOM FOREST REGRESSOR")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, max_depth=None,
                               random_state=42, n_jobs=-1)
rfr.fit(X_cal_tr, y_cal_tr)
pred_rfr = rfr.predict(X_cal_te)
rmse, mae, r2 = rapport_regression("RFR", y_cal_te, pred_rfr)
reg_results["RandomForest_R"] = r2
print(f"  [California] RMSE={rmse:.4f} | R2={r2:.4f}")

feat_imp_rfr = rfr.feature_importances_
feat_names_cal = california.feature_names

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg_scatter(y_cal_te, pred_rfr, "Random Forest Regressor -- California", axes[0])
idx_rfr = np.argsort(feat_imp_rfr)[::-1]
axes[1].barh([feat_names_cal[i] for i in idx_rfr], feat_imp_rfr[idx_rfr],
              color="#e74c3c")
axes[1].set_title("Feature Importances -- RFR (California)", fontweight="bold")
plt.suptitle("RANDOM FOREST REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 15. GRADIENT BOOSTING REGRESSOR -- Dataset : California Housing
# ============================================================
print("\n" + "=" * 60)
print("15. GRADIENT BOOSTING REGRESSOR")
print("=" * 60)

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                  max_depth=4, random_state=42,
                                  subsample=0.8)
gbr.fit(X_cal_tr, y_cal_tr)
pred_gbr = gbr.predict(X_cal_te)
rmse, mae, r2 = rapport_regression("GBR", y_cal_te, pred_gbr)
reg_results["GradientBoosting_R"] = r2
print(f"  [California] RMSE={rmse:.4f} | R2={r2:.4f}")

train_staged = [r2_score(y_cal_te, pred_s) for pred_s in gbr.staged_predict(X_cal_te)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg_scatter(y_cal_te, pred_gbr, "GBR -- California Housing", axes[0])
axes[1].plot(np.arange(1, gbr.n_estimators+1), train_staged, "b-", lw=1.5)
axes[1].set_xlabel("Iterations"); axes[1].set_ylabel("R2 (test)")
axes[1].set_title("Convergence GBR -- California", fontweight="bold")
axes[1].grid(True)
plt.suptitle("GRADIENT BOOSTING REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 16. SUPPORT VECTOR REGRESSION (SVR) -- Dataset : Diabetes
# ============================================================
print("\n" + "=" * 60)
print("16. SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 60)

from sklearn.svm import SVR

svr_pipe = Pipeline([("sc", StandardScaler()),
                      ("svr", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))])
svr_pipe.fit(X_dia_tr, y_dia_tr)
pred_svr = svr_pipe.predict(X_dia_te)
rmse, mae, r2 = rapport_regression("SVR", y_dia_te, pred_svr)
reg_results["SVR"] = r2
print(f"  [Diabetes] SVR RBF | RMSE={rmse:.2f} | R2={r2:.4f}")

# Comparaison kernels SVR
kernels_svr = ["linear", "poly", "rbf"]
r2s_svr = []
for k in kernels_svr:
    svr_k = Pipeline([("sc", StandardScaler()),
                       ("svr", SVR(kernel=k, C=100))])
    svr_k.fit(X_dia_tr, y_dia_tr)
    r2s_svr.append(r2_score(y_dia_te, svr_k.predict(X_dia_te)))
    print(f"  SVR kernel={k:6s} | R2={r2s_svr[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg_scatter(y_dia_te, pred_svr, "SVR (RBF) -- Diabetes", axes[0])
axes[1].bar(kernels_svr, r2s_svr, color=["#3498db", "#2ecc71", "#e74c3c"])
axes[1].set_title("Comparaison kernels SVR -- Diabetes", fontweight="bold")
axes[1].set_xlabel("Kernel"); axes[1].set_ylabel("R2")
axes[1].set_ylim(0, 1.05)
plt.suptitle("SUPPORT VECTOR REGRESSION (SVR)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 17. KNN REGRESSOR -- Dataset : California Housing (subset)
# ============================================================
print("\n" + "=" * 60)
print("17. KNN REGRESSOR")
print("=" * 60)

from sklearn.neighbors import KNeighborsRegressor

# Utiliser un sous-ensemble pour la rapidite
idx_sub = np.random.default_rng(42).choice(len(X_cal_tr), 3000, replace=False)
X_cal_sub = X_cal_tr[idx_sub]; y_cal_sub = y_cal_tr[idx_sub]

k_range_r = range(1, 21)
r2s_knnr  = []
for k in k_range_r:
    knnr = Pipeline([("sc", StandardScaler()),
                      ("knn", KNeighborsRegressor(n_neighbors=k))])
    knnr.fit(X_cal_sub, y_cal_sub)
    r2s_knnr.append(r2_score(y_cal_te, knnr.predict(X_cal_te)))

knnr_best = Pipeline([("sc", StandardScaler()),
                       ("knn", KNeighborsRegressor(n_neighbors=7))])
knnr_best.fit(X_cal_sub, y_cal_sub)
pred_knnr = knnr_best.predict(X_cal_te)
rmse, mae, r2 = rapport_regression("KNNR", y_cal_te, pred_knnr)
reg_results["KNN_R"] = r2
print(f"  [California] KNN(k=7) | R2={r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg_scatter(y_cal_te, pred_knnr, "KNN Regressor (k=7) -- California", axes[0])
axes[1].plot(list(k_range_r), r2s_knnr, "bo-", lw=2)
axes[1].axvline(7, color="red", linestyle="--", label="k=7 selectionne")
axes[1].set_xlabel("k voisins"); axes[1].set_ylabel("R2")
axes[1].set_title("R2 vs k -- KNN Regressor", fontweight="bold")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("KNN REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# 18. MLP REGRESSOR -- Dataset : California Housing
# ============================================================
print("\n" + "=" * 60)
print("18. MLP REGRESSOR (Neural Network)")
print("=" * 60)

from sklearn.neural_network import MLPRegressor

mlp_reg = Pipeline([
    ("sc",  StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001
    ))
])
mlp_reg.fit(X_cal_tr, y_cal_tr)
pred_mlp_reg = mlp_reg.predict(X_cal_te)
rmse, mae, r2 = rapport_regression("MLPReg", y_cal_te, pred_mlp_reg)
reg_results["MLP_R"] = r2
print(f"  [California] MLP(128-64-32) | RMSE={rmse:.4f} | R2={r2:.4f}")

loss_reg = mlp_reg["mlp"].loss_curve_
val_reg  = mlp_reg["mlp"].validation_scores_

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_reg_scatter(y_cal_te, pred_mlp_reg, "MLP Regressor -- California", axes[0])
ax2 = axes[1].twinx()
axes[1].plot(loss_reg, "b-", lw=1.5, label="Train Loss")
ax2.plot(val_reg, "r-", lw=1.5, label="Val R2")
axes[1].set_xlabel("Iterations"); axes[1].set_ylabel("Loss (bleu)", color="blue")
ax2.set_ylabel("Validation R2 (rouge)", color="red")
axes[1].set_title("Courbes d'entrainement MLP Reg. -- California", fontweight="bold")
axes[1].grid(True)
plt.suptitle("MLP REGRESSOR", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ============================================================
# COMPARAISON GLOBALE REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("COMPARAISON GLOBALE -- REGRESSION")
print("=" * 60)

from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm           import SVR
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

regressors = {
    "LinearReg"         : Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge(a=1)"        : Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso(a=0.1)"      : Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1, max_iter=5000))]),
    "ElasticNet"        : Pipeline([("sc", StandardScaler()), ("m", ElasticNet(max_iter=5000))]),
    "DecisionTree"      : Pipeline([("sc", StandardScaler()), ("m", DecisionTreeRegressor(max_depth=6, random_state=42))]),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR (rbf)"         : Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=100))]),
    "KNN (k=7)"         : Pipeline([("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=7))]),
    "MLP Reg."          : Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))]),
}

names_r, rmses_r, r2s_r = [], [], []
for name, reg in regressors.items():
    reg.fit(X_dia_tr, y_dia_tr)   # Tous sur Diabetes pour comparaison equitable
    pred = reg.predict(X_dia_te)
    rmse = np.sqrt(mean_squared_error(y_dia_te, pred))
    r2   = r2_score(y_dia_te, pred)
    names_r.append(name); rmses_r.append(rmse); r2s_r.append(r2)
    print(f"{name:22s} | RMSE : {rmse:.2f} | R2 : {r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
colors_r = plt.cm.tab10(np.linspace(0, 1, len(names_r)))
bars_r = axes[0].barh(names_r, r2s_r, color=colors_r)
axes[0].set_xlabel("R2 (Diabetes, test)")
axes[0].set_title("R2 Score -- Tous les regresseurs (Diabetes)", fontweight="bold")
axes[0].set_xlim(0, 1.05)
for bar, r2 in zip(bars_r, r2s_r):
    axes[0].text(r2 + 0.01, bar.get_y() + bar.get_height()/2,
                  f"{r2:.3f}", va="center", fontsize=8)

bars_r2 = axes[1].barh(names_r, rmses_r, color=colors_r)
axes[1].set_xlabel("RMSE (Diabetes, test)")
axes[1].set_title("RMSE -- Tous les regresseurs (Diabetes)", fontweight="bold")

plt.suptitle("COMPARAISON GLOBALE -- REGRESSION (Diabetes)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

print("\n[OK] Script termine -- tous les algorithmes de prediction executes avec succes.")
