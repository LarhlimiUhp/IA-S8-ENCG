# Compte Rendu — Algorithmes de Prédiction Supervisée (Partie 3)

## Méthodes complémentaires : Classifieurs linéaires, Régression robuste, GLM

> **Fichier source :** `prediction_examples_3.py`  
> **Date :** Mars 2026

---

## Table des matières

**CLASSIFICATION**

1. [Ridge Classifier](#1-ridge-classifier)
2. [Naive Bayes Multinomial](#2-naive-bayes-multinomial)
3. [Naive Bayes Bernoulli](#3-naive-bayes-bernoulli)
4. [Perceptron](#4-perceptron)
5. [Passive-Aggressive Classifier](#5-passive-aggressive-classifier)
6. [Calibration de modèles](#6-calibration-de-modèles)

**RÉGRESSION**

1. [Régression Polynomiale](#7-régression-polynomiale)
2. [Régression Quantile](#8-régression-quantile)
3. [Régression Isotonique](#9-régression-isotonique)
4. [RANSAC Regressor](#10-ransac-regressor)
5. [Theil-Sen Regressor](#11-theil-sen-regressor)
6. [Tweedie Regressor (GLM)](#12-tweedie-regressor-glm)

7. [Synthèse globale](#13-synthèse-globale)

---

## 1. Ridge Classifier

### Principe

Ridge Classifier convertit la classification en un problème de **régression Ridge** : chaque classe est encodée comme une colonne de la matrice indicatrice one-hot, puis Ridge est appliqué. La classe prédite est celle dont la sortie de régression est la plus élevée.

```
argmax_k  wk · x + bk      avec   wk = solution Ridge sur colonne k
```

**Avantage :** Plus rapide que la Régression Logistique (solution analytique, pas d'itérations). **RidgeClassifierCV** sélectionne automatiquement alpha par CV interne.

```python
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

rc = Pipeline([
    ("sc", StandardScaler()),
    ("rc", RidgeClassifier(alpha=1.0))  # alpha = regularisation L2
])
rc.fit(X_iris_tr, y_iris_tr)

# Selection automatique de alpha
rcv = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0])
rcv.fit(X_iris_tr_s, y_iris_tr)
print(f"Alpha optimal : {rcv.alpha_}")
```

| Dataset | Accuracy |
|---------|---------|
| Iris | **~97%** |

---

## 2. Naive Bayes Multinomial

### Principe

MultinomialNB modélise des **données de comptage** (ex.: fréquences de mots) :

```
P(xi | y=k) = (Nki + α) / (Nk + α × p)    # lissage de Laplace
```

Couplé à **TF-IDF** (Term Frequency – Inverse Document Frequency), c'est le classifieur de texte le plus utilisé.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("mnb",   MultinomialNB(alpha=0.1))  # alpha = lissage de Laplace
])
pipeline.fit(train_texts, train_labels)

# Features les plus discriminantes par classe
log_probs = pipeline["mnb"].feature_log_prob_  # (n_classes, n_features)
```

| Dataset | Accuracy |
|---------|---------|
| 20 Newsgroups (4 catégories) | **~90–93%** |

---

## 3. Naive Bayes Bernoulli

### Principe

BernoulliNB modélise des **features binaires** (0/1 — présence/absence) :

```
P(xi=1 | y=k) = pki       P(xi=0 | y=k) = 1 − pki
```

Utile pour les données texte binaires (mot présent ou non).

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer

bnb = Pipeline([
    ("bin", Binarizer(threshold=0.0)),  # binarise les features
    ("bnb", BernoulliNB(alpha=1.0))
])
```

| Dataset | GaussianNB | BernoulliNB |
|---------|-----------|------------|
| Breast Cancer | ~94% | **~88–92%** |

---

## 4. Perceptron

### Principe

Le Perceptron est le plus simple des classificateurs linéaires. Mise à jour uniquement sur les **exemples mal classés** :

```
si  yi(w·xi) ≤ 0 :   w ← w + η·yi·xi
```

| Propriétés | Valeurs |
|-----------|---------|
| Convergence garantie | Seulement si données **linéairement séparables** |
| Pas de probabilités | `decision_function` uniquement |
| Très rapide | O(n×p) par epoch |

```python
from sklearn.linear_model import Perceptron

perc = Pipeline([
    ("sc",   StandardScaler()),
    ("perc", Perceptron(
        max_iter=1000, eta0=0.1,  # learning rate
        tol=1e-4, random_state=42
    ))
])
# partial_fit() : apprentissage en ligne (incremental)
```

---

## 5. Passive-Aggressive Classifier

### Principe

PA Classifier est un **algorithme en ligne** qui met à jour le modèle uniquement quand il fait une erreur, avec une mise à jour **minimale** pour corriger l'erreur :

| Hinge loss (PA) | Hinge² loss (PA-II) |
|-----------------|---------------------|
| Mise à jour conservative | Correction plus agressive |

- **Passif** : modèle inchangé si prédiction correcte
- **Agressif** : mise à jour minimale pour corriger les erreurs

```python
from sklearn.linear_model import PassiveAggressiveClassifier

pa = Pipeline([
    ("sc",  StandardScaler()),
    ("pag", PassiveAggressiveClassifier(
        C=1.0,        # agressivite (grand = plus de mises a jour)
        max_iter=1000,
        random_state=42
    ))
])
```

---

## 6. Calibration de modèles

### Principe

Un modèle est **bien calibré** si P̂(y=1|x) = 0.7 signifie que 70% des points avec cette confiance sont réellement positifs.

**Courbe de calibration :** fraction de positifs observés vs probabilité prédite moyenne.

**Méthodes de calibration :**

| Méthode | Description | Quand l'utiliser |
|---------|-------------|-----------------|
| **Platt Scaling (sigmoid)** | Ajuste une sigmoïde sur les sorties | Faibles données d'ajustement |
| **Isotonic** | Régression isotonique non-paramétrique | Plus de données disponibles |

**Brier Score** = MSE des probabilités (0=parfait, 1=pire)

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibrer un SVM (qui ne donne pas de probas par defaut)
calibrated_svm = CalibratedClassifierCV(
    SVC(kernel="rbf"),
    cv=5,
    method="sigmoid"  # ou "isotonic"
)
calibrated_svm.fit(X_train_s, y_train)
proba = calibrated_svm.predict_proba(X_test_s)

# Verifier la calibration
frac_pos, mean_pred = calibration_curve(y_test, proba[:, 1], n_bins=10)
brier = brier_score_loss(y_test, proba[:, 1])
```

---

## 7. Régression Polynomiale

### Principe

Étend la régression linéaire en ajoutant des **features polynomiales** :

```
deg=2  :   [x1, x2] → [x1, x2, x1², x1·x2, x2²]
```

```
ŷ = w0 + w1x1 + w2x2 + w3x1² + w4x1x2 + w5x2²
```

**Explosion combinatoire :** p=10 features, degré=2 → 65 features ; degré=3 → 285 features.
→ Ajouter **régularisation Ridge** pour les degrés > 1.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

poly_pipe = Pipeline([
    ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=1.0))   # regularisation necessaire !
])
```

| Degré | Nb features (p=10) | R² (Diabetes) |
|-------|-------------------|-------------|
| 1 | 10 | ~0.48 |
| **2** | **65** | **~0.52** |
| 3 | 285 | ~0.45 (overfitting) |

---

## 8. Régression Quantile

### Principe

La régression quantile estime le **quantile τ** de la distribution conditionnelle de y|x (au lieu de la moyenne) :

```
minimiser  Σ ρτ(yi - ŷi)
ρτ(r) = r·τ  si r≥0 ,  r·(τ−1) si r<0
```

**Avantages :**

- Robuste aux outliers (pas de carré)
- Fournit des **intervalles de prédiction** asymétriques
- Ne suppose pas la normalité des résidus

```python
from sklearn.linear_model import QuantileRegressor

# Estimation de la mediane (Q50) : equivalent a la regression MAE
q50 = QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs")

# Intervalle de prediction [Q10, Q90] ~ 80% de couverture
q10 = QuantileRegressor(quantile=0.1, alpha=0.001, solver="highs")
q90 = QuantileRegressor(quantile=0.9, alpha=0.001, solver="highs")
```

---

## 9. Régression Isotonique

### Principe

La régression isotonique contraint la fonction prédite à être **monotone croissante** (ou décroissante) — résout le problème de pool adjacent (PAVA) :

```
minimiser  Σ (yi - ŷi)²    sous    ŷ1 ≤ ŷ2 ≤ ... ≤ ŷn
```

Utile pour :

- Calibration de probabilités
- Relations dose-réponse
- Post-traitement d'une prédiction ordinale

```python
from sklearn.isotonic import IsotonicRegression

# Fonctionne uniquement sur 1 feature -> utiliser PCA d'abord
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(X_1d_train, y_train)
pred = iso.predict(X_1d_test)  # prediction garantie monotone
```

---

## 10. RANSAC Regressor

### Principe

RANSAC (*Random Sample Consensus*) détecte et exclut les **outliers structurels** :

1. Tirer aléatoirement un sous-ensemble minimal d'exemples
2. Entraîner le modèle sur ce sous-ensemble
3. Calculer les résidus pour tous les exemples
4. Les exemples avec |résidu| < seuil = inliers
5. Répéter, conserver le modèle avec le plus d'inliers

```python
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(
    estimator=LinearRegression(),
    min_samples=0.5,         # au moins 50% d'inliers
    residual_threshold=50,   # |residu| < 50 -> inlier
    max_trials=200,
    random_state=42
)
ransac.fit(X_train_s, y_train)
pred = ransac.predict(X_test_s)

# Masque des inliers detectes
inliers = ransac.inlier_mask_
```

| Modèle | R² avec 30 outliers |
|--------|---------------------|
| OLS | ~0.10–0.20 |
| **RANSAC** | **~0.40–0.45** |

---

## 11. Theil-Sen Regressor

### Principe

Estime les coefficients comme la **médiane des pentes** calculées sur toutes les paires d'exemples :

```
β̂ = médiane{ (yj - yi)/(xj - xi) : i < j }
```

**Robustesse :** Résistant jusqu'à ~29.3% d'outliers.
**Inconvénient :** O(n²) ou O(n³) — lent sur grands datasets (`max_subpopulation` pour limiter).

```python
from sklearn.linear_model import TheilSenRegressor

ts = TheilSenRegressor(
    max_subpopulation=1000,  # limite le nombre de paires evaluees
    random_state=42
)
```

### Comparaison des régresseurs robustes (Diabetes + outliers)

| Méthode | R² (sans outliers) | R² (avec 30 outliers) |
|---------|------------|---------------------|
| OLS | ~0.48 | ~0.10 |
| Huber | ~0.47 | ~0.45 |
| TheilSen | ~0.46 | ~0.43 |
| **RANSAC** | **~0.46** | **~0.44** |

---

## 12. Tweedie Regressor (GLM)

### Principe

Le Tweedie Regressor généralise plusieurs distributions via le paramètre `power` :

| power | Distribution | Usage |
|-------|-------------|-------|
| 0 | **Gaussienne** | Régression standard |
| 1 | **Poisson** | Comptage (sinistres, clics) |
| 2 | **Gamma** | Montants (coûts, durées) |
| (1,2) | **Tweedie** | Comptage + montant (assurance) |

**Liens :**

- `identity` pour Gaussienne
- `log` pour Poisson/Gamma/Tweedie (assure y > 0)

```python
from sklearn.linear_model import TweedieRegressor

tw = TweedieRegressor(
    power=1,          # Poisson (y doit etre > 0)
    alpha=0.5,        # regularisation
    link="log",       # lien log pour Poisson/Gamma
    max_iter=1000
)
tw.fit(X_train_s, np.clip(y_train, 0.01, None))
```

---

## 13. Synthèse globale

### Classification — Tous algorithmes (Iris)

| Rang | Algorithme | Test Acc | CV 5-fold |
|------|-----------|---------|---------|
| 1 | MLP (256-128) | ~100% | ~97% |
| 2 | SVM RBF | ~97% | ~97% |
| 3 | Random Forest | ~97% | ~96% |
| 4 | Gradient Boosting | ~97% | ~97% |
| 5 | Extra Trees | ~97% | ~96% |
| … | LogReg, LDA, QDA | ~97% | ~96% |
| bas | Perceptron | Variable | ~90% |

### Régression — Tous algorithmes (Diabetes)

| Rang | Algorithme | R² | RMSE |
|------|-----------|-----|------|
| 1 | **Gradient Boosting** | **~0.57** | ~50 |
| 2 | Random Forest | ~0.55 | ~51 |
| 3 | MLP Regressor | ~0.54 | ~52 |
| 4 | Poly(deg=2)+Ridge | ~0.52 | ~53 |
| … | Ridge, BayesRidge, SVR | ~0.49 | ~54 |
| bas | Isotonic (1D) | ~0.35 | ~61 |

### Choisir le bon algorithme

```
CLASSIFICATION :
  Interprétabilité  -> LDA, DecisionTree, LogReg
  Texte             -> MultinomialNB + TF-IDF
  Grande vitesse    -> SGD, Perceptron, RidgeClassifier
  Performance max.  -> GBM, SVM, MLP, Stacking/Voting

REGRESSION :
  Linéaire          -> LinearReg, Ridge, Lasso, ElasticNet
  Non-linéaire      -> Random Forest, GBM
  Outliers          -> Huber, RANSAC, TheilSen
  Intervalles       -> Quantile Regression, Bayesian Ridge
  Comptage (>0)     -> Tweedie (Poisson/Gamma)
  Monotonie         -> Isotonic Regression
```

---

## Bilan des 3 parties

| Partie | Fichiers | Algorithmes |
|--------|---------|------------|
| **Partie 1** | `prediction_examples.py` + `prediction_rapport.md` | LR, DT, RF, GB, SVM, KNN, NB, AdaBoost, MLP, ExtraTrees + 8 regresseurs |
| **Partie 2** | `prediction_examples_2.py` + `prediction_rapport_2.md` | LDA, QDA, SGD, Bagging, Voting, Stacking, GP + 8 regresseurs + GridSearch + CV |
| **Partie 3** | `prediction_examples_3.py` + `prediction_rapport_3.md` | RidgeClf, MNB, BNB, Perceptron, PA, Calibration + Poly, Quantile, Isotonic, RANSAC, TheilSen, Tweedie + **Synthèse** |

**Total : ~40 algorithmes supervisés couverts**

---

*Mars 2026 — Habilitation 2026*
