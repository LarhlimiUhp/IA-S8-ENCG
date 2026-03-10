# Compte Rendu — Algorithmes de Prédiction Supervisée (Partie 2)

## Méthodes avancées : Ensembles, Bayésien, Processus Gaussiens, Optimisation

> **Fichier source :** `prediction_examples_2.py`  
> **Date :** Mars 2026  
> **Datasets :** Iris, Breast Cancer, Wine, Digits, Diabetes, California Housing

---

## Table des matières

**CLASSIFICATION (suite)**

1. [LDA — Analyse Discriminante Linéaire](#1-analyse-discriminante-linéaire-lda)
2. [QDA — Analyse Discriminante Quadratique](#2-analyse-discriminante-quadratique-qda)
3. [SGD Classifier](#3-sgd-classifier)
4. [Bagging Classifier](#4-bagging-classifier)
5. [Voting Classifier](#5-voting-classifier-hard--soft)
6. [Stacking Classifier](#6-stacking-classifier)
7. [Gaussian Process Classifier](#7-gaussian-process-classifier)

**RÉGRESSION (suite)**

1. [Bayesian Ridge Regression](#8-bayesian-ridge-regression)
2. [Huber Regressor](#9-huber-regressor)
3. [ARD Regression](#10-ard-regression)
4. [PLS Regression](#11-partial-least-squares-pls)
5. [SGD Regressor](#12-sgd-regressor)
6. [Bagging Regressor](#13-bagging-regressor)
7. [Stacking Regressor](#14-stacking-regressor)
8. [Gaussian Process Regressor](#15-gaussian-process-regressor)

**OPTIMISATION**

1. [GridSearchCV](#16-gridsearchcv)
2. [RandomizedSearchCV](#17-randomizedsearchcv)
3. [Cross-Validation avancée](#18-cross-validation-avancée)

---

# SECTION A — CLASSIFICATION (suite)

---

## 1. Analyse Discriminante Linéaire (LDA)

### Principe théorique

LDA cherche les projections qui **maximisent la séparation inter-classes** tout en **minimisant la variance intra-classe**.

**Critère de Fisher :**

```
J(w) = (w^T SB w) / (w^T SW w)
```

- **SB** : matrice de dispersion inter-classes (Between-class scatter)
- **SW** : matrice de dispersion intra-classe (Within-class scatter)

La solution est donnée par les vecteurs propres de SW⁻¹ SB.

**Double usage :**

- **Classificateur** : assigne la classe à la gaussienne la plus probable
- **Réducteur de dimension** : projette sur au plus k−1 axes discriminants

| LDA | PCA |
|-----|-----|
| Supervisé (utilise les étiquettes) | Non supervisé |
| Maximise séparation inter-classes | Maximise variance globale |
| k−1 composantes max | Min(n, p) composantes |

### Code commenté

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_iris_tr, y_iris_tr)
pred = lda.predict(X_iris_te)

# LDA comme reducteur de dimension : projection 4D -> 2D (k-1=2 pour Iris)
X_iris_lda = lda.transform(iris.data)  # (150, 2)
# Les 2 axes LD1 et LD2 separent visuellement les 3 classes

# Voir les scalings (directions discriminantes)
print(lda.scalings_)   # (n_features, n_components)
```

### Résultats attendus

| Dataset | Accuracy |
|---------|---------|
| Iris | **~97%** |
| Wine | **~98-100%** |

---

## 2. Analyse Discriminante Quadratique (QDA)

### Principe théorique

QDA est une extension de LDA qui permet des **matrices de covariance différentes par classe** (frontières quadratiques au lieu de linéaires).

**Modèle par classe :**

```
P(x | y=k) = N(x | μk, Σk)
```

**Frontière de décision :**

```
log P(y=k | x) = -½ log|Σk| - ½(x-μk)^T Σk⁻¹ (x-μk) + log πk
```

→ Expression quadratique en x (parabole/ellipse en 2D)

| LDA | QDA |
|-----|-----|
| Σ unique pour toutes les classes | Σk distinct par classe |
| Frontières linéaires | Frontières quadratiques |
| Moins de paramètres | Plus de paramètres (risque overfitting) |
| Robuste avec peu de données | Nécessite plus de données par classe |

**Paramètre `reg_param`** ∈ [0,1] : régularisation des matrices de covariance (0=QDA pur, 1=LDA)

### Code commenté

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis(
    reg_param=0.1   # regularisation : evite la singularite des matrices Sigma_k
)
qda.fit(X_iris_tr, y_iris_tr)
pred = qda.predict(X_iris_te)

# Probabilites posterieures : P(classe | x) pour chaque echantillon
proba = qda.predict_proba(X_iris_te)  # (n_test, n_classes)

# Parametres appris : une moyenne et une covariance par classe
print(qda.means_)      # (n_classes, n_features)
print(qda.covariance_) # (n_classes, n_features, n_features)
```

### Résultats attendus

| Dataset | LDA | QDA |
|---------|-----|-----|
| Iris | ~97% | **~97%** |
| Wine | ~98% | **~98%** |

---

## 3. SGD Classifier

### Principe théorique

SGDClassifier implémente la **descente de gradient stochastique** pour entraîner des modèles linéaires à grande échelle. C'est une implémentation générique qui supporte plusieurs fonctions de perte.

**Mise à jour SGD :**

```
w ← w − η × [∂L(y, w·x)/∂w + α × ∂R(w)/∂w]
```

- η = learning rate
- L = fonction de perte
- R = régularisation (L1, L2, ElasticNet)

| Fonction de perte | Modèle équivalent | Usage |
|------------------|-------------------|-------|
| `hinge` | SVM linéaire | Classification robuste |
| `log_loss` | Régression Logistique | Probabilités |
| `modified_huber` | Robuste + probabilités | Données bruitées |
| `perceptron` | Perceptron classique | Simple |

**Avantage majeur :** Efficace sur **très grands datasets** (en ligne/out-of-core avec `partial_fit`)

### Code commenté

```python
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([
    ("sc",  StandardScaler()),        # OBLIGATOIRE : SGD sensible aux echelles
    ("sgd", SGDClassifier(
        loss="log_loss",              # -> regression logistique via SGD
        penalty="l2",                 # regularisation Ridge
        alpha=0.0001,                 # intensite de regularisation
        max_iter=1000,                # nb max d'epochs
        random_state=42,
        n_jobs=-1
    ))
])
sgd.fit(X_dig_tr, y_dig_tr)

# SGD supporte l'apprentissage en ligne (incremental)
# Pour de nouveaux lots de donnees :
# sgd["sgd"].partial_fit(X_new, y_new, classes=np.unique(y))
```

### Résultats attendus

| Dataset | Loss | Accuracy |
|---------|------|---------|
| Digits | hinge | **~94%** |
| Digits | log_loss | **~94%** |
| Digits | modified_huber | **~93%** |

---

## 4. Bagging Classifier

### Principe théorique

Bagging (*Bootstrap Aggregating*) entraîne plusieurs classifieurs indépendants sur des **sous-ensembles bootstrap** du dataset et agrège leurs prédictions par vote.

**Algorithme :**

1. Pour t = 1 à T :
   - Tirer un bootstrap sample Bt (n exemples avec remise)
   - Entraîner un classifieur ht sur Bt
2. Prédiction : ŷ = mode({ h1(x), h2(x), ..., hT(x) })

**Propriété :**

```
Var(h̄) ≤ Var(h) / T       (si les h_t sont non corrélés)
```

Le bagging réduit la **variance** du modèle de base.

**Out-Of-Bag (OOB) :** Les exemples non inclus dans Bt servent de validation interne → estimation gratuite de la performance.

| Paramètre | Rôle |
|----------|------|
| `n_estimators` | Nb de classifieurs de base |
| `max_samples` | Fraction d'exemples par base learner |
| `max_features` | Fraction de features par base learner |
| `bootstrap` | True = avec remise (classique) |
| `oob_score` | Activer le score OOB |

### Code commenté

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),  # arbres profonds
    n_estimators=100,
    max_samples=0.8,       # 80% des exemples -> diversite
    max_features=0.8,      # 80% des features -> diversite supplementaire
    bootstrap=True,        # avec remise
    oob_score=True,        # score OOB gratuit
    random_state=42,
    n_jobs=-1
)
bag.fit(X_wine_tr, y_wine_tr)

# OOB score : estimation de la generalisation sans besoin du test set
print(f"OOB Score : {bag.oob_score_:.4f}")

# Note : Random Forest = BaggingClassifier(estimator=DT) avec max_features="sqrt"
# La difference principale : RF choisit les features a chaque NOEUD, pas par arbre
```

### Résultats attendus

| Méthode | Wine Accuracy |
|---------|-------------|
| DT seul | ~82% |
| **Bagging (100 DT)** | **~97%** |
| OOB Score | ~96% |

---

## 5. Voting Classifier (Hard & Soft)

### Principe théorique

VotingClassifier combine les prédictions de plusieurs classifieurs **hétérogènes** (de natures différentes).

**Hard Voting :** La classe finale est le **mode** des prédictions individuelles.

```
ŷ = argmax_k Σt 1[ht(x) = k]
```

**Soft Voting :** La classe finale maximise la **somme des probabilités** estimées.

```
ŷ = argmax_k Σt P̂t(y=k | x)
```

> Soft Voting est généralement **meilleur** que Hard Voting car il pondère les votes par la confiance.

**Condition de succès :** Les classifieurs membres doivent faire des **erreurs différentes** (faible corrélation des erreurs).

| Paramètre | Rôle |
|----------|------|
| `voting` | `hard` ou `soft` |
| `weights` | Optionnel : pondération de chaque classifieur |
| `flatten_transform` | Pour accéder aux probabilités individuelles |

### Code commenté

```python
from sklearn.ensemble import VotingClassifier

# Membres heterogenes (modeles de natures differentes)
base_clfs = [
    ("lr",  Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])),
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("svm", Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", probability=True))])),
]

# Soft Voting : necessite probability=True pour tous les membres
vote_soft = VotingClassifier(estimators=base_clfs, voting="soft")
vote_soft.fit(X_iris_tr, y_iris_tr)

# Probabilites du Voting (moyenne des probas membres)
proba = vote_soft.predict_proba(X_iris_te)  # (n_test, n_classes)
```

### Résultats attendus (Iris)

| Méthode | Accuracy |
|---------|---------|
| Individuel (best) | ~97% |
| Hard Voting | **~97%** |
| **Soft Voting** | **~97–100%** |

---

## 6. Stacking Classifier

### Principe théorique

Stacking (Stacked Generalization) est une méthode d'ensemble à **deux niveaux** :

- **Niveau 0** : base learners entraînés sur les données originales
- **Niveau 1** : meta-learner entraîné sur les **prédictions hors-échantillon** des base learners

**Procédure (avec cv=5) :**

```
Pour chaque fold i (1..5) :
    Entraîner les base learners sur les folds 1..5 \ {i}
    Prédire sur le fold i -> génère des features pour le meta-learner

Meta-learner entraîné sur ces prédictions hors-échantillon
```

**Avantage vs Voting :** Le meta-learner **apprend** comment combiner les base learners, au lieu d'une simple moyenne.

| Paramètre | Rôle |
|----------|------|
| `cv` | Nb de folds pour générer les prédictions OOS |
| `stack_method` | `predict_proba` (mieux) ou `predict` |
| `passthrough` | True : passe aussi X original au meta-learner |
| `final_estimator` | Meta-learner (LR est souvent suffisant) |

### Code commenté

```python
from sklearn.ensemble import StackingClassifier

stack = StackingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=50, random_state=42)),
        ("gb",  GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ("knn", Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))])),
        ("nb",  GaussianNB()),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,                         # 5 folds pour generer les features OOS
    stack_method="predict_proba", # features = probabilites par classe
    passthrough=False             # le meta-learner NE VOIT PAS X directement
)
stack.fit(X_bc_tr, y_bc_tr)
pred = stack.predict(X_bc_te)
```

### Résultats attendus

| Dataset | Stacking | Meilleur base learner |
|---------|---------|----------------------|
| Breast Cancer | **~97–98%** | ~97% |

---

## 7. Gaussian Process Classifier

### Principe théorique

Le GPC modélise la distribution posterior sur les fonctions de décision via un **Processus Gaussien** et utilise l'inférence approximative (Laplace) pour la classification.

**Processus Gaussien :** Distribution sur les fonctions, entièrement caractérisée par une fonction de **covariance (kernel)** :

```
f(x) ~ GP(m(x), k(x,x'))
```

**Kernels courants :**

| Kernel | Formule | Forme des frontières |
|--------|---------|---------------------|
| **RBF** | exp(-||x-x'||²/2l²) | Lisses, infiniment différentiables |
| **Matern** | (fonction de Bessel) | Moins lisses, plus réalistes |
| **DotProduct** | x·x' + σ² | Linéaires |

**Avantage :** Fournit des **probabilités bien calibrées** et quantifie l'incertitude.
**Inconvénient :** Complexité O(n³) — limité aux datasets < ~5000 exemples.

### Code commenté

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern

gpc = GaussianProcessClassifier(
    kernel=RBF(length_scale=1.0),   # kernel RBF gaussien
    random_state=42
)
gpc.fit(X_iris_tr_s, y_iris_tr)   # normaliser avant GPC

# Probabilites posterieures bien calibrees
proba = gpc.predict_proba(X_iris_te_s)  # (n_test, n_classes)

# Kernel optimise par maximisation de la log-vraisemblance marginale
print(gpc.kernel_)  # length_scale optimise automatiquement
```

### Résultats attendus (Iris)

| Kernel | Accuracy |
|--------|---------|
| RBF | **~97%** |
| Matern | **~97%** |

---

# SECTION B — RÉGRESSION (suite)

---

## 8. Bayesian Ridge Regression

### Principe théorique

La régression Ridge Bayésienne place des **priors gaussiens** sur les poids et estime la distribution posterior complète (pas seulement le MAP).

**Modèle :**

```
y = Xw + ε,    ε ~ N(0, λ⁻¹)
w ~ N(0, α⁻¹ I)
```

**Prédiction :** Distribution posterior predictive N(μ_pred, σ²_pred) — fournit une **incertitude** sur chaque prédiction.

**Avantage clé :** Les hyperparamètres α (précision des poids) et λ (précision du bruit) sont **estimés automatiquement** par Evidence Maximization — pas besoin de GridSearch.

### Code commenté

```python
from sklearn.linear_model import BayesianRidge

bay_ridge = BayesianRidge(max_iter=500)
bay_ridge.fit(X_dia_tr_s, y_dia_tr)

# Prediction avec incertitude : retourne mean ET std de prediction
pred, pred_std = bay_ridge.predict(X_dia_te_s, return_std=True)

# pred_std[i] = ecart-type de la prediction pour le point i
# -> intervalle de confiance : [pred - 2*std, pred + 2*std] ~ 95%
# Points dans des regions peu denses du train -> std plus grande

# Hyperparametres estimes automatiquement
print(f"alpha_ (precision poids) : {bay_ridge.alpha_:.4f}")
print(f"lambda_ (precision bruit): {bay_ridge.lambda_:.4f}")
```

### Résultats attendus (Diabetes)

| Métrique | Valeur |
|----------|--------|
| R² | **~0.49** |
| RMSE | **~54** |
| Incertitude moyenne (std) | **~40–50** |

---

## 9. Huber Regressor

### Principe théorique

Le Huber Regressor utilise la **fonction de perte de Huber** — un compromis entre MSE (quadratique) et MAE (linéaire) qui rend le modèle robuste aux outliers.

**Fonction de perte de Huber :**

```
L_ε(r) = r²/2          si |r| ≤ ε
          ε(|r| - ε/2)  si |r| > ε
```

→ Quadratique près de 0 (précis pour les petits résidus), linéaire loin de 0 (non-influencé par les outliers).

| OLS (MSE) | Huber | MAE |
|-----------|-------|-----|
| Très sensible aux outliers | Robuste modéré | Robuste (mais non différentiable en 0) |
| Convergence rapide | Bon compromis | Convergence lente |

**Paramètre `epsilon`** : frontière entre régime quadratique (résidus < ε) et linéaire (résidus > ε). ε=1.35 → 95% d'efficacité relative vs OLS sur données normales.

### Code commenté

```python
from sklearn.linear_model import HuberRegressor

# Simuler des outliers
y_train_noisy = y_train.copy()
y_train_noisy[outlier_idx] += 300   # perturbation massive

# OLS est effondre par les outliers
ols_noisy = LinearRegression().fit(X_train_s, y_train_noisy)
r2_ols = r2_score(y_test, ols_noisy.predict(X_test_s))  # << normal

# Huber est robuste : les outliers ont peu d'influence
huber = HuberRegressor(
    epsilon=1.35,   # frontiere robustesse/precision (1.35 = recommande)
    max_iter=500
)
huber.fit(X_train_s, y_train_noisy)
r2_hub = r2_score(y_test, huber.predict(X_test_s))   # beaucoup plus proche du R2 sans outliers

# huber.outliers_ : masque booleen des exemples traites comme outliers
print(f"Outliers detectes : {huber.outliers_.sum()}")
```

### Résultats attendus (Diabetes avec outliers)

| Modèle | R² (sans outliers) | R² (avec outliers) |
|--------|-------------------|-------------------|
| OLS | ~0.48 | **~0.10–0.20** (effondré) |
| **Huber** | ~0.47 | **~0.45** (robuste) |

---

## 10. ARD Regression

### Principe théorique

ARD (*Automatic Relevance Determination*) est une extension de la régression Ridge Bayésienne avec un **hyperparamètre distinct par feature** (λj) :

```
P(w) = N(0, diag(α₁⁻¹, α₂⁻¹, ..., αp⁻¹))
```

Chaque αj est estimé séparément → les features peu pertinentes reçoivent un très grand αj, ce qui annule automatiquement leur coefficient (sparse solution).

**Différence avec Ridge/Bayesian Ridge :**

| Ridge | Bayesian Ridge | ARD |
|-------|--------------|-----|
| Un α global | Un α global estimé | Un αj par feature |
| Tous les coefficients réduits également | Tous réduits également | Coefficients non pertinents → 0 |

### Code commenté

```python
from sklearn.linear_model import ARDRegression

ard = ARDRegression(max_iter=500)
ard.fit(X_dia_tr_s, y_dia_tr)
pred = ard.predict(X_dia_te_s)

# lambda_ : un par feature (precisement : precision des poids)
# 1/lambda_[j] = variance du poids wj = pertinence de la feature j
# lambda_ tres grand -> coefficient -> 0 -> feature non pertinente
lambda_ = ard.lambda_         # (n_features,)
relevance = 1.0 / lambda_     # features avec grande pertinence ~ Lasso

# Coefficients appris (proches de 0 pour features non pertinentes)
print(ard.coef_)
```

---

## 11. Partial Least Squares (PLS)

### Principe théorique

PLS cherche des composantes qui **maximisent la covariance** entre X et y (plutôt que la variance de X seul comme PCA).

**Décomposition :**

```
X = T P^T + E     T = X W*    (scores, loadings, résidus)
y = T q^T + f
```

PLS est particulièrement utile quand :

- Les features sont **fortement corrélées** (multicolinéarité)
- n < p (plus de features que d'exemples)
- Contexte chimiométrique/spectroscopique

| PCA | PCR | PLS |
|-----|-----|-----|
| Max Var(X) | Max Var(X), puis régressi. | Max Cov(X, y) |
| Non supervisé | Semi-supervisé | Supervisé par y |

### Code commenté

```python
from sklearn.cross_decomposition import PLSRegression

# Trouver le nombre optimal de composantes
r2_scores = []
for n in range(1, X_train_s.shape[1]+1):
    pls = PLSRegression(n_components=n)
    pls.fit(X_train_s, y_train)
    r2_scores.append(r2_score(y_test, pls.predict(X_test_s)))
best_n = np.argmax(r2_scores) + 1

pls = PLSRegression(n_components=best_n)
pls.fit(X_train_s, y_train)
pred = pls.predict(X_test_s).ravel()  # .ravel() : shape (n,) pas (n,1)

# Scores et loadings
T = pls.x_scores_     # Scores (n, n_components) : projection de X
P = pls.x_loadings_   # Loadings (n_features, n_components)
q = pls.y_loadings_   # Loadings de y
```

### Résultats attendus (Diabetes)

| n_composantes | R² |
|--------------|-----|
| 1 | ~0.35 |
| **5** | **~0.48** |
| 10 | ~0.48 |

---

## 12. SGD Regressor

### Principe théorique

Idem au SGD Classifier mais avec des **fonctions de perte de régression** : MSE (Ridge), MAE (Lasso), Huber, Epsilon-Insensitive (SVR).

**Avantage décisif :** `partial_fit()` permet l'apprentissage incrémental sur flux de données massifs.

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    loss="squared_error",   # MSE -> equivalent Ridge via SGD
    penalty="l2",           # regularisation Ridge
    alpha=0.001,
    max_iter=1000,
    random_state=42
)
sgd_reg.fit(X_train_s, y_train)

# Learning online : nouveau batch de donnees toutes les N heures
# for X_batch, y_batch in stream:
#     sgd_reg.partial_fit(X_batch, y_batch)
```

---

## 13. Bagging Regressor

### Principe théorique

Identique au Bagging Classifier mais la prédiction finale est la **moyenne** (et non le vote) des prédictions de base :

```
ŷ(x) = (1/T) Σt ht(x)
```

La réduction de variance est particulièrement bénéfique pour les **modèles à haute variance** (arbres non élagués).

```python
from sklearn.ensemble import BaggingRegressor

bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=6),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bag_reg.fit(X_cal_tr_s, y_cal_tr)
pred = bag_reg.predict(X_cal_te_s)
print(f"OOB R2 : {bag_reg.oob_score_:.4f}")
```

### Résultats attendus (California)

| Méthode | R² |
|---------|-----|
| DT seul (depth=6) | ~0.67 |
| **Bagging (100 DT)** | **~0.78** |

---

## 14. Stacking Regressor

### Principe théorique

Extension du Stacking Classifier à la régression — les base learners prédisent des valeurs numériques, et le meta-learner (souvent Ridge) apprend à combiner ces prédictions :

```
x*_i = [ĥ1(xi), ĥ2(xi), ..., ĥT(xi)]    (features pour le meta-learner)
ŷ_meta = Ridge(x*_i)
```

```python
from sklearn.ensemble import StackingRegressor

stack_reg = StackingRegressor(
    estimators=[
        ("rf",  RandomForestRegressor(n_estimators=50, random_state=42)),
        ("gb",  GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ("dtr", DecisionTreeRegressor(max_depth=6)),
    ],
    final_estimator=Ridge(alpha=1.0),   # meta-learner lineaire
    cv=5,
    passthrough=False,
    n_jobs=-1
)
stack_reg.fit(X_cal_tr_s, y_cal_tr)
pred = stack_reg.predict(X_cal_te_s)
```

### Résultats attendus (California)

| Méthode | R² |
|---------|-----|
| RF seul | ~0.80 |
| GB seul | ~0.83 |
| **Stacking** | **~0.82–0.84** |

---

## 15. Gaussian Process Regressor

### Principe théorique

Le GPR fournit une **distribution posterior complète** sur les prédictions, quantifiant l'incertitude point par point.

**Processus Gaussien :**

```
f(x) ~ GP(m(x), k(x,x'))
```

**Posterior conditionnellement aux observations :**

```
f* | X, y, x* ~ N(μ*, Σ*)
μ* = k(x*,X) [k(X,X) + σ²I]⁻¹ y
Σ* = k(x*,x*) - k(x*,X) [k(X,X) + σ²I]⁻¹ k(X,x*)
```

**Kernels courants en régression :**

| Kernel | Formule | Usage |
|--------|---------|-------|
| **RBF** | exp(-||x-x'||²/2l²) | Fonctions très lisses |
| **Matern** | ... | Plus robuste, moins lisse |
| **WhiteKernel** | σ²_n δ(x,x') | Bruit observationnel |
| **C × RBF** | amplitude × forme | Contrôle l'amplitude |

### Code commenté

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Kernel composite : amplitude * RBF + bruit
kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=3,   # plusieurs restarts pour trouver le meilleur kernel
    random_state=42
)
gpr.fit(X_dia_tr_s, y_dia_tr)

# Prediction avec incertitude :
pred, std = gpr.predict(X_dia_te_s, return_std=True)
# IC 95% : [pred - 2*std, pred + 2*std]

# Kernel optimise automatiquement (length_scale, amplitude, bruit)
print(gpr.kernel_)   # montre les hyperparametres optimises
```

### Résultats attendus (Diabetes)

| Métrique | Valeur |
|----------|--------|
| R² | **~0.48–0.52** |
| Rapport au GPR | IC ~95% bien calibrés |

---

# SECTION C — OPTIMISATION DES HYPERPARAMÈTRES

---

## 16. GridSearchCV

### Principe théorique

GridSearchCV effectue une **recherche exhaustive** sur une grille de combinaisons d'hyperparamètres, évaluée par validation croisée.

**Nombre total de configurations :**

```
N_total = Π |Pi|    (produit des tailles de chaque grille)
```

**Combiné avec StratifiedKFold :**

```
N_ajustements = N_total × k_folds
```

| Paramètre | Rôle |
|----------|------|
| `param_grid` | Dictionnaire {param: [valeurs]} |
| `cv` | Stratégie de validation (entier ou objet CV) |
| `scoring` | Métrique à optimiser |
| `n_jobs` | Parallélisation (-1 = tous les cœurs) |
| `refit` | Re-entraîner sur tout le train avec les meilleurs params |

### Code commenté

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    "n_estimators"      : [50, 100, 200],
    "max_depth"         : [None, 5, 10],
    "max_features"      : ["sqrt", "log2"],
    "min_samples_split" : [2, 5],
}
# -> 3 x 3 x 2 x 2 = 36 combinaisons x 5 folds = 180 ajustements

gs = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
    refit=True           # re-entraine le meilleur modele sur tout le train
)
gs.fit(X_train, y_train)

# Meilleurs hyperparametres trouves
print(gs.best_params_)
print(gs.best_score_)    # CV score

# Le meilleur modele est directement disponible pour la prediction
pred = gs.predict(X_test)

# Analyser tous les resultats
import pandas as pd
df_results = pd.DataFrame(gs.cv_results_)
```

---

## 17. RandomizedSearchCV

### Principe théorique

RandomizedSearchCV échantillonne **aléatoirement** n_iter combinaisons d'hyperparamètres parmi des **distributions continues ou discrètes**.

**Avantage lors d'un grand espace de recherche :**

- GridSearch : coût O(Π|Pi|)
- RandomSearch : coût O(n_iter) — **indépendant de la taille de la grille**
- Pour n_iter fixé, RS couvre plus d'espace que GS si les grilles sont grandes

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators"  : randint(50, 500),       # distribution uniforme discrete
    "learning_rate" : uniform(0.01, 0.3),     # distribution uniforme continue
    "max_depth"     : randint(2, 10),
    "subsample"     : uniform(0.6, 0.4),      # [0.6, 1.0]
}

rs = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=40,              # 40 combinaisons aleatoires (vs 1000 pour GridSearch)
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42
)
rs.fit(X_train_s, y_train)
print(rs.best_params_)
print(rs.best_score_)
```

---

## 18. Cross-Validation avancée

### Stratégies disponibles

| Stratégie | Usage |
|-----------|-------|
| **KFold** | Régression, données IID |
| **StratifiedKFold** | Classification (préserve les proportions) |
| **RepeatedKFold** | Estimation plus robuste (repeat k fois) |
| **LeaveOneOut** | Petits datasets (coûteux) |
| **GroupKFold** | Données groupées (sujets, patients) |
| **TimeSeriesSplit** | Données temporelles (no data leakage) |

```python
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                      cross_validate, LeaveOneOut)

# Validation croisee avec plusieurs metriques simultanees
cv_results = cross_validate(
    rf, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=["accuracy", "f1_macro", "roc_auc_ovr"],
    n_jobs=-1,
    return_train_score=True  # aussi les scores train (pour diagnostiquer over/underfitting)
)

print(f"Test  accuracy : {cv_results['test_accuracy'].mean():.4f}")
print(f"Train accuracy : {cv_results['train_accuracy'].mean():.4f}")
# Si train >> test -> surapprentissage

# Repeated StratifiedKFold : reduit la variance de l'estimation CV
rep_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores_rep = cross_val_score(rf, X, y, cv=rep_cv, scoring="accuracy")
# 5 x 3 = 15 scores -> moyenne plus stable qu'un simple KFold 5
```

---

## Synthèse des résultats

### Classification (Iris)

| Algorithme | Accuracy | Remarque |
|-----------|---------|---------|
| LDA | ~97% | Excellent + réduction dimensionnelle |
| QDA | ~97% | Frontières quadratiques |
| SGD (log_loss) | ~94% | Scalable, online learning |
| Bagging | ~97% | Variance ↓ |
| **Soft Voting** | **~97–100%** | Combine hétérogènes |
| Stacking | **~97–98%** | Meta-apprentissage |
| GPC | ~97% | Incertitude quantifiée |

### Régression (Diabetes)

| Algorithme | R² | Particularité |
|-----------|-----|-------------|
| Bayesian Ridge | ~0.49 | Incertitude sur chaque prédiction |
| **Stacking** | **~0.54** | Meilleur R² |
| **GPR** | **~0.50** | Distribution predictive complète |
| Huber | ~0.47 | Robuste aux outliers |
| PLS | ~0.48 | Multicolinéarité |
| ARD | ~0.48 | Feature selection automatique |

### Guide de l'optimisation

```
Budget temporel ?
├── Illimité → GridSearchCV (exhaustif)
├── Limité   → RandomizedSearchCV (n_iter=50-100)
└── Minuscule → RandomSearchCV (n_iter=20) ou Optuna/Hyperopt

Validation pour estimer les performances :
├── Dataset > 10k exemples → KFold 5
├── Dataset 1k-10k        → StratifiedKFold 5 (classif) / RepeatedKFold 5x3
├── Dataset < 1k          → RepeatedKFold 10x5 ou LeaveOneOut
└── Données temporelles   → TimeSeriesSplit (JAMAIS KFold standard)
```

---

## Dépendances

```bash
pip install scikit-learn scipy numpy matplotlib pandas
```

## Exécution

```bash
python prediction_examples_2.py
```

---

*Mars 2026 — Habilitation 2026*
