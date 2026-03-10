# Compte Rendu — Algorithmes de Prédiction Supervisée

## Exemples avec données réelles (scikit-learn)

> **Fichier source :** `prediction_examples.py`  
> **Date :** Mars 2026  
> **Datasets :** Iris (150×4), Breast Cancer (569×30), Wine (178×13), Digits (1797×64), California Housing (20 640×8), Diabetes (442×10)

---

## Table des matières

**CLASSIFICATION**

1. [Régression Logistique](#1-régression-logistique)
2. [Decision Tree Classifier](#2-decision-tree-classifier)
3. [Random Forest Classifier](#3-random-forest-classifier)
4. [Gradient Boosting Classifier](#4-gradient-boosting-classifier)
5. [Support Vector Machine (SVM)](#5-support-vector-machine-svm)
6. [K-Nearest Neighbors (KNN)](#6-k-nearest-neighbors-knn)
7. [Naive Bayes (GaussianNB)](#7-naive-bayes-gaussiannb)
8. [AdaBoost Classifier](#8-adaboost-classifier)
9. [Perceptron Multicouche (MLP)](#9-perceptron-multicouche-mlp)
10. [Extra Trees Classifier](#10-extra-trees-classifier)

**RÉGRESSION**
11. [Régression Linéaire (OLS)](#11-régression-linéaire-ols)
12. [Ridge / Lasso / ElasticNet](#12-ridge--lasso--elasticnet)
13. [Decision Tree Regressor](#13-decision-tree-regressor)
14. [Random Forest Regressor](#14-random-forest-regressor)
15. [Gradient Boosting Regressor](#15-gradient-boosting-regressor)
16. [Support Vector Regression (SVR)](#16-support-vector-regression-svr)
17. [KNN Regressor](#17-knn-regressor)
18. [MLP Regressor](#18-mlp-regressor)

1. [Comparaisons globales](#19-comparaisons-globales)
2. [Guide de choix algorithmique](#20-guide-de-choix-algorithmique)

---

## Introduction

### Types d'apprentissage supervisé

| Type | Cible (y) | Métrique principale |
|------|-----------|-------------------|
| **Classification** | Catégorie discrète | Accuracy, F1, AUC |
| **Régression** | Valeur continue | RMSE, MAE, R² |

### Métriques de classification

| Métrique | Formule | Interprétation |
|----------|---------|---------------|
| **Accuracy** | (VP+VN) / Total | Taux correct global |
| **Précision** | VP / (VP+FP) | Qualité des prédictions positives |
| **Rappel** | VP / (VP+FN) | Détection des vrais positifs |
| **F1-Score** | 2 × P×R / (P+R) | Compromis précision/rappel |
| **AUC-ROC** | Aire sous la courbe ROC | Séparabilité binaire |

### Métriques de régression

| Métrique | Formule | Interprétation |
|----------|---------|---------------|
| **RMSE** | √(Σ(y-ŷ)²/n) | Erreur quadratique (même unité que y) |
| **MAE** | Σ|y-ŷ|/n | Erreur moyenne absolue (robuste) |
| **R²** | 1 - SS_res/SS_tot | 1=parfait, 0=modèle nul, <0=pire que moyenne |

### Datasets réels

| Dataset | Taille | Features | Cible | Algorithmes |
|---------|--------|----------|-------|-------------|
| **Iris** | 150 | 4 morpho. | 3 espèces | LR, RF, SVM, KNN, NB, MLP |
| **Breast Cancer** | 569 | 30 cellulaires | Bénin/Malin | GB, AdaBoost, LR, RF |
| **Wine** | 178 | 13 chimiques | 3 vins | DT, ExtraTrees |
| **Digits** | 1797 | 64 pixels 8×8 | Chiffres 0-9 | SVM, MLP |
| **California Hous.** | 20 640 | 8 géographiques | Prix médian | LinReg, DT, RF, GBR, KNN, MLP |
| **Diabetes** | 442 | 10 médicaux | Progression | Ridge, Lasso, SVR, comparaison |

---

## Préparation commune

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Split 80/20 stratifié (classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline normalisation + modele (recommande pour eviter la fuite de donnees)
model = Pipeline([
    ("scaler", StandardScaler()),  # ajuste sur train uniquement
    ("clf",    MonClassifieur())
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

> **Pourquoi un Pipeline ?** Le `StandardScaler` est ajusté uniquement sur les données d'entraînement et appliqué sans ré-ajustement sur le test, évitant ainsi le *data leakage*.

---

# SECTION A — CLASSIFICATION

---

## 1. Régression Logistique

### Principe théorique

Malgré son nom, la régression logistique est un **algorithme de classification**. Elle modélise la probabilité d'appartenance à une classe via la fonction sigmoïde :

```
P(y=1 | x) = σ(w·x + b) = 1 / (1 + exp(-(w·x + b)))
```

**Extension multi-classe :** Softmax (one-vs-rest ou multinomial)

```
P(y=k | x) = exp(wk·x) / Σj exp(wj·x)
```

**Optimisation :** Maximisation de la log-vraisemblance (équivalent à minimiser l'entropie croisée) — résolu avec L-BFGS, Newton-CG ou SGD.

**Régularisation :** Terme L2 par défaut (`C` = inverse de l'intensité de régularisation)

| Hyperparamètre | Rôle | Valeur typique |
|---------------|------|---------------|
| `C` | Inverse régularisation L2 | 0.01 à 100 |
| `max_iter` | Convergence (augmenter si besoin) | 1000 |
| `solver` | Optimiseur | `lbfgs` (petits), `saga` (grands) |
| `multi_class` | Stratégie multi-classe | `ovr` ou `multinomial` |

### Code commenté

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline : normalisation obligatoire pour LR (features a la meme echelle)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(
        max_iter=1000,       # augmenter si ConvergenceWarning
        C=1.0,               # regularisation L2 (defaut)
        random_state=42
    ))
])

lr_pipe.fit(X_iris_tr, y_iris_tr)
pred = lr_pipe.predict(X_iris_te)

# Probabilites posterieures (utile pour seuillage)
proba = lr_pipe.predict_proba(X_iris_te)  # shape : (n_test, n_classes)

# Coefficients : importance de chaque feature par classe
coefs = lr_pipe["lr"].coef_  # (n_classes, n_features)
# |coef[k, j]| grand -> feature j tres influent pour classe k
```

### Résultats attendus

| Dataset | Accuracy | Remarque |
|---------|---------|---------|
| Iris (test 20%) | **~97%** | Quasi-parfait sur 30 samples |
| Breast Cancer | **~97%** | Excellent pour binaire linéaire |

> La régression logistique est souvent l'**algorithme de référence** (baseline) en classification : rapide, interprétable, efficace sur données linéairement séparables.

---

## 2. Decision Tree Classifier

### Principe théorique

Un arbre de décision partitionne récursivement l'espace des features par des **règles if/else**. À chaque nœud, il cherche le split qui minimise l'impureté.

**Critères d'impureté :**

| Critère | Formule | Usage |
|---------|---------|-------|
| **Gini** | 1 - Σ p²k | Défaut (rapide) |
| **Entropie** | -Σ pk log(pk) | Information gain |

**Sélection du meilleur split :**

```
Gain(nœud, feature j, seuil t) = Impureté(parent) - Σ (ni/n) × Impureté(enfanti)
```

**Risques :**

- **Sous-apprentissage :** arbre trop peu profond
- **Sur-apprentissage :** arbre trop profond (mémorise le bruit)
- **Solution :** `max_depth`, `min_samples_split`, post-pruning

| Hyperparamètre | Rôle |
|---------------|------|
| `max_depth` | Profondeur maximale (évite sur-apprentissage) |
| `min_samples_split` | Nb min de samples pour diviser un nœud |
| `min_samples_leaf` | Nb min de samples dans une feuille |
| `criterion` | `gini` ou `entropy` |

### Code commenté

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(
    max_depth=4,          # limite la profondeur -> generalisation
    min_samples_split=5,  # noeud avec < 5 points n'est pas divise
    random_state=42
)
dt.fit(X_wine_tr, y_wine_tr)
pred_dt = dt.predict(X_wine_te)

# Feature importances : somme des reductions d'impurete ponderees par nb de samples
importances = dt.feature_importances_  # shape : (n_features,)
# La feature avec la plus grande importance est utilisee pres de la racine

# Visualisation de l'arbre
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt,
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,       # couleur selon classe majoritaire
          rounded=True,
          fontsize=7, ax=ax)
```

### Résultats attendus

| Dataset | Accuracy | max_depth |
|---------|---------|----------|
| Wine (test 20%) | **~92–97%** | 4 |

> Les arbres sont très interprétables mais instables (sensibles à l'ordre des données). Random Forest corrige cela.

---

## 3. Random Forest Classifier

### Principe théorique

Random Forest est un **ensemble d'arbres de décision** entraînés par **bagging** (Bootstrap Aggregating) avec sélection aléatoire des features à chaque split.

**Mécanisme :**

1. Pour t = 1 à T :
   - Tirer un bootstrap sample (n exemples avec remise)
   - Entraîner un arbre sur ce sample avec m = √p features aléatoires à chaque nœud
2. Prédiction finale : vote majoritaire des T arbres

**Propriétés :**

- **Variance réduite** par rapport à un seul arbre
- **Robuste** aux outliers et au sur-apprentissage
- **Feature importance** = réduction moyenne de Gini sur tous les arbres
- **Out-of-Bag (OOB) error** : estimation interne sur les exemples non tirés

| Hyperparamètre | Rôle | Valeur typique |
|---------------|------|---------------|
| `n_estimators` | Nb d'arbres (plus = meilleur, plus lent) | 100–500 |
| `max_features` | Features par nœud | `sqrt` (classif), `log2` |
| `max_depth` | `None` = arbres complets | None |
| `oob_score` | Activer le score OOB | True |

### Code commenté

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

# Pas de normalisation requise (arbres : invariant aux echelles)
rf = RandomForestClassifier(
    n_estimators=100,       # 100 arbres
    max_features="sqrt",    # sqrt(n_features) features par noeud
    random_state=42,
    n_jobs=-1               # parallelisation sur tous les coeurs
)
rf.fit(X_iris_tr, y_iris_tr)
pred_rf = rf.predict(X_iris_te)

# Feature importances : moyennees sur tous les arbres
importances = rf.feature_importances_

# Learning Curve : voir si underfitting ou overfitting
train_sizes, train_sc, test_sc = learning_curve(
    rf, X_iris_scaled, iris.target,
    cv=5,                          # 5-fold cross-validation
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy"
)
# Si train_sc >> test_sc : overfitting
# Si les deux sont bas et stagnent : underfitting
```

### Résultats attendus

| Dataset | Accuracy | CV 5-fold |
|---------|---------|---------|
| Iris | **~97%** | ~96% ± 2% |
| Breast Cancer | **~97%** | ~97% ± 1% |

> Random Forest est souvent le meilleur rapport **performance/complexité** parmi les classifieurs classiques.

---

## 4. Gradient Boosting Classifier

### Principe théorique

Gradient Boosting construit les arbres **séquentiellement** : chaque nouvel arbre corrige les erreurs des précédents en suivant le gradient de la fonction de perte.

**Algorithme général :**

```
F0(x) = argmin_γ Σ L(yi, γ)    # initialisation
Fm(x) = F(m-1)(x) + η × hm(x)  # ajout d'un arbre hm
```

où hm est entraîné sur les **pseudo-résidus** : rm = -∂L/∂F(m-1)

**Différences avec Random Forest :**

| Random Forest | Gradient Boosting |
|--------------|-------------------|
| Arbres parallèles | Arbres séquentiels |
| Bagging (bootstrap) | Boosting (résidus) |
| Variance ↓ | Biais ↓ |
| Robuste overfitting | Sensible à η et n_estimators |

| Hyperparamètre | Rôle | Impact |
|---------------|------|--------|
| `n_estimators` | Nb d'arbres | Plus = plus puissant, risque surapprentissage |
| `learning_rate` | Taux d'apprentissage η | Petit + beaucoup d'arbres = meilleure generalisation |
| `max_depth` | Profondeur (2–5 recommandé) | Controle la complexite |
| `subsample` | Fraction d'exemples par arbre | <1 = stochastic GBM |

### Code commenté

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,    # 200 arbres sequentiels
    learning_rate=0.1,   # eta : petit = meilleure generalisation
    max_depth=3,         # arbres peu profonds (stumps a 3 niveaux)
    subsample=0.8,       # stochastic GBM : 80% des exemples par arbre
    random_state=42
)
gb.fit(X_bc_tr, y_bc_tr)
pred = gb.predict(X_bc_te)

# Probabilites (classification binaire)
proba = gb.predict_proba(X_bc_te)[:, 1]  # proba classe positive

# Courbe de deviance (perte) par iteration -> detecte les GB overfit
train_score = gb.train_score_  # perte training a chaque iteration

# staged_predict : predictions apres chaque arbre -> courbe d'apprentissage
staged_acc = [accuracy_score(y_bc_te, p) for p in gb.staged_predict(X_bc_te)]
```

### Résultats attendus

| Dataset | Accuracy |
|---------|---------|
| Breast Cancer | **~97–98%** |

> Gradient Boosting est souvent le **meilleur algorithme single-model** (sans neural nets) — mais plus lent à entraîner et plus sensible aux hyperparamètres.

---

## 5. Support Vector Machine (SVM)

### Principe théorique

SVM cherche l'**hyperplan à marge maximale** qui sépare les classes. La marge est la distance entre l'hyperplan et les **vecteurs supports** (points les plus proches).

**Problème d'optimisation (cas séparable) :**

```
minimiser  (1/2)||w||²
sous        yi(w·xi + b) ≥ 1 pour tout i
```

**Kernel Trick :** SVM peut opérer dans des espaces de haute dimension sans les calculer explicitement via un kernel k(xi, xj) :

| Kernel | Formule | Usage |
|--------|---------|-------|
| **Linear** | x·z | Données linéairement séparables |
| **RBF** | exp(-γ||x-z||²) | Défaut, polyvalent |
| **Polynomial** | (γx·z + r)^d | Patterns polynomiaux |
| **Sigmoïde** | tanh(γx·z + r) | Similaire à NN |

**SVM doux (Soft Margin) :** paramètre C contrôle le compromis marge/erreurs

| C grand | C petit |
|---------|---------|
| Marge étroite, peu d'erreurs | Grande marge, accepte erreurs |
| Risque sur-apprentissage | Risque sous-apprentissage |

### Code commenté

```python
from sklearn.svm import SVC

# SVM avec kernel RBF (le plus general)
svm_pipe = Pipeline([
    ("sc",  StandardScaler()),   # OBLIGATOIRE : SVM sensible aux echelles
    ("svm", SVC(
        kernel="rbf",            # kernel gaussien (par defaut)
        C=1.0,                   # penalite d'erreur (regularisation inverse)
        gamma="scale",           # gamma = 1/(n_features * Var(X)) -> adaptatif
        probability=False,       # True pour predict_proba (plus lent)
        random_state=42
    ))
])

svm_pipe.fit(X_iris_tr, y_iris_tr)
pred = svm_pipe.predict(X_iris_te)

# SVM sur Digits (haute dimension : efficace grace au kernel trick)
svm_digits = Pipeline([
    ("sc",  StandardScaler()),
    ("svm", SVC(kernel="rbf", C=5.0, gamma="scale"))
])
svm_digits.fit(X_dig_tr, y_dig_tr)
pred_dig = svm_digits.predict(X_dig_te)
# Attendu : accuracy > 98% sur Digits !
```

### Résultats attendus

| Dataset | Kernel | Accuracy |
|---------|--------|---------|
| Iris | RBF | **~97%** |
| Iris | Linear | **~97%** |
| Iris | Poly | **~97%** |
| Digits | RBF | **>98%** |

> SVM avec kernel RBF est excellent en haute dimension. Sur Digits (64 features), il surpasse souvent les arbres.

---

## 6. K-Nearest Neighbors (KNN)

### Principe théorique

KNN est un algorithme **non-paramétrique paresseux** (lazy learning) : il ne construit pas de modèle explicite mais mémorise tout le dataset d'entraînement.

**Prédiction :** pour un point x, trouver les k plus proches voisins dans l'ensemble d'entraînement, puis :

- **Classification :** vote majoritaire des k voisins
- **Régression :** moyenne des valeurs des k voisins

**Distance par défaut :** Euclidienne (Minkowski avec p=2)

```
d(xi, xj) = √(Σ (xik - xjk)²)
```

**Choix de k :**

- k petit → frontières complexes, surapprentissage
- k grand → frontières lisses, sous-apprentissage
- Souvent k = √n, impair pour éviter les ex-aequo

| Hyperparamètre | Rôle |
|---------------|------|
| `n_neighbors` | k (plus important) |
| `weights` | `uniform` (vote égal) ou `distance` (voisins proches = plus de poids) |
| `metric` | Distance utilisée (Euclidean, Manhattan, etc.) |

### Code commenté

```python
from sklearn.neighbors import KNeighborsClassifier

# Recherche du k optimal
for k in range(1, 21):
    knn = Pipeline([
        ("sc",  StandardScaler()),      # OBLIGATOIRE : KNN est sensible aux echelles
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    knn.fit(X_iris_tr, y_iris_tr)
    acc = accuracy_score(y_iris_te, knn.predict(X_iris_te))
    # Tracer acc vs k : maximum souvent entre k=3 et k=7

# Modele final
knn_final = Pipeline([
    ("sc",  StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,         # k=5 : bon compromis en general
        weights="distance",    # voisins proches = plus influents
        metric="euclidean"
    ))
])
knn_final.fit(X_iris_tr, y_iris_tr)
```

### Résultats attendus

| Dataset | k | Accuracy |
|---------|---|---------|
| Iris | 5 | **~97%** |

> KNN est simple et efficace pour de petits datasets. Il devient lent en production (O(n) prédictions) — utiliser des k-d trees ou ball trees pour les grands datasets.

---

## 7. Naive Bayes (GaussianNB)

### Principe théorique

Naive Bayes applique le **théorème de Bayes** avec l'hypothèse (naïve) d'**indépendance conditionnelle** des features :

```
P(y | x1, ..., xp) ∝ P(y) × Π P(xi | y)
```

**GaussianNB :** modélise P(xi | y) par une gaussienne N(μky, σ²ky)

```
P(xi | y=k) = (1/√(2πσ²ki)) × exp(-(xi - μki)² / (2σ²ki))
```

**Avantages :**

- Très rapide (O(n×p) entraînement)
- Fonctionne bien avec peu de données
- Fournit des probabilités bien calibrées
- Robuste aux features non pertinentes

**Inconvénient principal :** l'hypothèse d'indépendance est souvent violée.

| Variante | Usage |
|---------|-------|
| GaussianNB | Features continues |
| MultinomialNB | Comptage (texte, NLP) |
| BernoulliNB | Features binaires |
| ComplementNB | Texte déséquilibré |

### Code commenté

```python
from sklearn.naive_bayes import GaussianNB

# Naive Bayes ne necessite pas de normalisation
# (il estime ses propres mu/sigma par classe)
gnb = GaussianNB()
gnb.fit(X_iris_tr, y_iris_tr)
pred = gnb.predict(X_iris_te)

# Probabilites posterieures : P(classe | x)
proba = gnb.predict_proba(X_iris_te)  # (n_test, 3) pour Iris

# Parametres appris : mu et sigma de chaque feature par classe
mu    = gnb.theta_   # (n_classes, n_features) : moyennes
sigma = gnb.var_     # (n_classes, n_features) : variances

# Exemple : mu[0] = moyenne des 4 features pour la classe 0 (Setosa)
print(f"Moyenne feature 1 (Setosa) : {mu[0, 0]:.3f}")
```

### Résultats attendus

| Dataset | Accuracy |
|---------|---------|
| Iris | **~97%** |
| Breast Cancer | **~94%** |

> Malgré son hypothèse simpliste, GaussianNB obtient de très bons résultats sur Iris et Breast Cancer. Il est surtout utile quand les données sont rares.

---

## 8. AdaBoost Classifier

### Principe théorique

AdaBoost (Adaptive Boosting) combine de **nombreux classifieurs faibles** (typiquement des stumps = arbres à 1 niveau) en leur attribuant des poids selon leurs performances.

**Algorithme (classification binaire) :**

1. Initialiser poids uniformes w_i = 1/n
2. Pour t = 1 à T :
   - Entraîner classifieur faible ht sur les poids {wi}
   - Calculer erreur pondérée : εt = Σ wi × 1[ht(xi) ≠ yi]
   - Calculer importance : αt = (1/2) × ln((1−εt)/εt)
   - Mettre à jour : wi ← wi × exp(−αt × yi × ht(xi))
   - Renormaliser les poids
3. Prédiction finale : signe(Σ αt × ht(x))

**Intuition :** Les exemples mal classés reçoivent plus de poids → les classifieurs suivants se concentrent dessus.

| Hyperparamètre | Rôle |
|---------------|------|
| `n_estimators` | Nb de classifieurs faibles |
| `learning_rate` | Réduit la contribution de chaque classifieur |
| `algorithm` | `SAMME` (discret) ou `SAMME.R` (probabilités) |

### Code commenté

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=100,          # 100 stumps
    learning_rate=0.5,         # contribution reduite -> meilleure generalisation
    algorithm="SAMME",         # SAMME : version robuste pour multi-classe
    random_state=42
)
ada.fit(X_bc_tr, y_bc_tr)
pred = ada.predict(X_bc_te)

# Evolution de l'accuracy au fur et a mesure des iterations
staged_acc = [accuracy_score(y_bc_te, pred)
              for pred in ada.staged_predict(X_bc_te)]
# staged_predict(X) : generateur -> pred apres k classifieurs (k=1..T)
# Utile pour choisir le nb optimal d'estimateurs

# Poids des classifieurs faibles : estimator_weights_[i]
# Plus alpha_i est grand, plus le classifieur i est important
```

### Résultats attendus

| Dataset | Accuracy |
|---------|---------|
| Breast Cancer | **~97%** |

---

## 9. Perceptron Multicouche (MLP)

### Principe théorique

Le MLP est un réseau de neurones artificiels composé de couches de **neurones entièrement connectés**.

**Architecture :**

```
Input (p) → Couche 1 (n1) → Couche 2 (n2) → ... → Output (k classes)
```

**Neurone :** z = σ(w·x + b) avec σ = fonction d'activation

| Activation | Formule | Usage |
|-----------|---------|-------|
| **ReLU** | max(0, x) | Couches cachées (défaut) |
| **Sigmoid** | 1/(1+e^−x) | Sortie binaire |
| **Softmax** | exp(xi)/Σexp(xj) | Sortie multi-classe |
| **Tanh** | (e^x−e^−x)/(e^x+e^−x) | RNN, alternative ReLU |

**Rétropropagation :** Calcul du gradient via la règle de chaîne, puis mise à jour par Adam/SGD.

**Early Stopping :** Arrêter si la validation ne s'améliore plus → évite le surapprentissage.

| Hyperparamètre | Rôle |
|---------------|------|
| `hidden_layer_sizes` | Architecte (ex: (256, 128, 64)) |
| `activation` | Fonction d'activation (`relu`, `tanh`) |
| `solver` | Optimiseur (`adam`, `sgd`, `lbfgs`) |
| `alpha` | Régularisation L2 |
| `early_stopping` | Arrêt anticipé |

### Code commenté

```python
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

mlp = Pipeline([
    ("sc",  StandardScaler()),      # OBLIGATOIRE : les gradients explosent sans normalisation
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3 couches cachees : 256, 128, 64 neurones
        activation="relu",                  # ReLU : evite la disparition du gradient
        solver="adam",                      # Adam : adaptatif, converge vite
        max_iter=300,                       # nb max d'epochs
        random_state=42,
        early_stopping=True,               # surveille la validation a chaque epoch
        validation_fraction=0.1            # 10% du train = monitored validation
    ))
])
mlp.fit(X_dig_tr, y_dig_tr)

# Courbes d'entrainement
loss_curve = mlp["mlp"].loss_curve_         # perte train par iteration
val_scores  = mlp["mlp"].validation_scores_ # accuracy val par iteration

# Poids de la 1ere couche : interpretables sur Digits (8x8 images)
weights_l1 = mlp["mlp"].coefs_[0]   # (64 features, n1 neurones)
# Visualiser w.reshape(8,8) pour voir les "recepteurs" appris
```

### Résultats attendus

| Dataset | Architecture | Accuracy |
|---------|-------------|---------|
| Digits (10 classes) | (256, 128, 64) | **~98%** |

> Le MLP est excellent sur des données structurées en haute dimension (Digits). Utiliser PyTorch/TensorFlow pour des architectures plus complexes (CNN, RNN).

---

## 10. Extra Trees Classifier

### Principe théorique

Extra Trees (Extreme Randomized Trees) est similaire à Random Forest mais avec une **randomisation encore plus forte** : les seuils de split sont choisis **aléatoirement** (pas optimisés).

**Différences avec Random Forest :**

| Random Forest | Extra Trees |
|--------------|-------------|
| Bootstrap sample | Tout le dataset |
| Meilleur seuil parmi m features | Seuil ALÉATOIRE parmi m features |
| Plus lent | Plus rapide |
| Variance légèrement plus élevée | Variance plus faible |

**Avantage :** Plus rapide que Random Forest (pas d'optimisation des seuils) tout en ayant une variance similaire ou inférieure grâce à la forte randomisation.

### Code commenté

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(
    n_estimators=200,    # plus d'arbres = meilleur (extra trees convergent vite)
    random_state=42,
    n_jobs=-1            # parallelisation
)
et.fit(X_wine_tr, y_wine_tr)   # pas de normalisation necessaire
pred = et.predict(X_wine_te)

# Feature importances (memes proprietes que RF)
importances = et.feature_importances_
```

### Résultats attendus

| Dataset | Accuracy |
|---------|---------|
| Wine (test 20%) | **~97–100%** |

---

# SECTION B — RÉGRESSION

---

## 11. Régression Linéaire (OLS)

### Principe théorique

La régression linéaire prédit une valeur continue via une **combinaison linéaire des features** :

```
ŷ = w0 + w1×x1 + w2×x2 + ... + wp×xp = Xw
```

**Méthode des Moindres Carrés Ordinaires (OLS) :**

```
minimiser  (1/n)||y - Xw||²
```

**Solution analytique :**

```
w* = (X^T X)^{-1} X^T y
```

**Hypothèses du modèle linéaire :**

1. Linéarité : E[y|x] = Xw
2. Homoscédasticité : Var[ε] = σ² constant
3. Indépendance des résidus
4. Normalité des résidus (pour les intervalles de confiance)

| Métrique | Ce qu'elle mesure |
|----------|-------------------|
| **R²** | Part de variance expliquée |
| **RMSE** | Erreur quadratique (même unité que y) |
| **MAE** | Erreur absolue (robuste aux outliers) |

### Code commenté

```python
from sklearn.linear_model import LinearRegression

lr_reg = Pipeline([
    ("sc", StandardScaler()),    # standardisation pour stabilite numerique
    ("lr", LinearRegression())   # OLS : solution analytique (X^T X)^-1 X^T y
])
lr_reg.fit(X_cal_tr, y_cal_tr)
pred = lr_reg.predict(X_cal_te)

# Coefficients : biais et poids
bias  = lr_reg["lr"].intercept_       # w0
coefs = lr_reg["lr"].coef_            # (n_features,)

# Residus : diagnostic de qualite du modele
residuals = y_cal_te - pred
# Residus aleatoires autour de 0 -> bon ajustement
# Patron dans les residus -> relation non-lineaire manquante

# Metriques
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2   = r2_score(y_cal_te, pred)
rmse = np.sqrt(mean_squared_error(y_cal_te, pred))
mae  = mean_absolute_error(y_cal_te, pred)
```

### Résultats attendus

| Dataset | R² | RMSE |
|---------|-----|------|
| California Housing | **~0.59** | ~0.71 |
| Diabetes | **~0.48** | ~55 |

> La régression linéaire est simple et rapide. Elle échoue sur des relations non-linéaires (R² < 0.6 souvent signe de non-linéarité).

---

## 12. Ridge / Lasso / ElasticNet

### Principe théorique

Ces trois modèles ajoutent une **pénalité de régularisation** à la régression linéaire pour réduire le sur-apprentissage et gérer la multicolinéarité.

| Modèle | Fonction de perte | Effet |
|--------|------------------|-------|
| **Ridge (L2)** | MSE + α × Σwi² | Rétrécit tous les coefficients |
| **Lasso (L1)** | MSE + α × Σ|wi| | Rend certains coefficients exactement 0 (sélection) |
| **ElasticNet** | MSE + α(ρ Σ|wi| + (1-ρ) Σwi²) | Combine L1 et L2 |

**Lasso :** Idéal pour la sélection de variables (sparse models)
**Ridge :** Idéal quand toutes les features sont pertinentes
**ElasticNet :** Compromis (utile si features corrélées entre elles)

**Chemin de régularisation Lasso :** Quand α ↑, les coefficients sont progressivement annulés → révèle les features les plus importantes.

### Code commenté

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge : L2 shrinkage
ridge = Ridge(alpha=1.0)   # alpha=0 -> LinearRegression
ridge.fit(X_dia_tr_s, y_dia_tr)
pred_ridge = ridge.predict(X_dia_te_s)

# Lasso : L1 sparse (feature selection)
lasso = Lasso(alpha=0.1, max_iter=5000)
# ATTENTION : max_iter a augmenter pour alpha faible (convergence plus lente)
lasso.fit(X_dia_tr_s, y_dia_tr)
print(f"Features zeroes : {(lasso.coef_ == 0).sum()}")  # features eliminées

# ElasticNet : combine L1 + L2
enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
# l1_ratio=0 -> Ridge, l1_ratio=1 -> Lasso

# Chemin de regularisation Lasso : montrer comment les coefs evoluent avec alpha
alphas = np.logspace(-3, 2, 50)
coef_paths = [Lasso(alpha=a, max_iter=10000).fit(X_dia_tr_s, y_dia_tr).coef_
              for a in alphas]
# Tracer coef_paths vs log(alpha) : les features disparaissent progressivement
```

### Résultats attendus (Diabetes)

| Modèle | R² | RMSE |
|--------|-----|------|
| OLS (pas de reg.) | ~0.48 | ~55 |
| **Ridge (α=1)** | **~0.49** | ~54 |
| **Lasso (α=0.1)** | **~0.48** | ~55 |
| **ElasticNet** | **~0.48** | ~55 |

> Sur Diabetes, les performances sont similaires car les features sont déjà bien sélectionnées. Lasso devient déterminant sur des datasets avec des centaines/milliers de features.

---

## 13. Decision Tree Regressor

### Principe théorique

Le Decision Tree Regressor partitionne l'espace en **régions rectangulaires** et prédit la **moyenne** de la variable cible dans chaque région.

**Critère de split (régression) :**

```
MSE_reduction = MSE(parent) - (nL/n × MSE(gauche) + nR/n × MSE(droite))
```

**Comportement :**

- `max_depth=None` → mémorise exactement (R²=1 sur train, overfitting)
- `max_depth` petit → sous-apprentissage
- `max_depth=6` → souvent bon compromis

### Code commenté

```python
from sklearn.tree import DecisionTreeRegressor

# Etude de l'effet max_depth
for d in [2, 4, 6, 8, None]:
    dtr = DecisionTreeRegressor(max_depth=d, random_state=42)
    dtr.fit(X_cal_tr, y_cal_tr)
    r2_train = r2_score(y_cal_tr, dtr.predict(X_cal_tr))
    r2_test  = r2_score(y_cal_te, dtr.predict(X_cal_te))
    print(f"depth={str(d):4s} | Train R2={r2_train:.3f} | Test R2={r2_test:.3f}")
# None -> Train R2=1.0, Test R2=~0.60 (overfitting)
# depth=6 -> Train R2=~0.75, Test R2=~0.67 (meilleur test)
```

### Résultats attendus

| max_depth | Train R² | Test R² (California) |
|----------|---------|---------------------|
| 2 | 0.49 | 0.49 |
| 4 | 0.67 | 0.64 |
| **6** | **0.75** | **0.67** |
| None | 1.00 | 0.60 (overfitting) |

---

## 14. Random Forest Regressor

### Principe théorique

Random Forest Regressor applique le même principe d'ensemble que pour la classification, mais la prédiction finale est la **moyenne** des prédictions de tous les arbres :

```
F(x) = (1/T) × Σt ft(x)
```

La **réduction de variance** due à l'agrégation améliore systématiquement les performances par rapport à un seul arbre.

### Code commenté

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,       # arbres complets (la RF se regularise par l'ensemble)
    max_features=1/3,     # 1/3 des features par noeud (recommande pour regression)
    random_state=42,
    n_jobs=-1
)
rfr.fit(X_cal_tr, y_cal_tr)
pred = rfr.predict(X_cal_te)

# Feature importances : tres utile pour comprendre les predicteurs cles
feat_imp = rfr.feature_importances_
# California Housing : MedInc (revenu median) est de loin la feature la plus importante
```

### Résultats attendus

| Dataset | R² | RMSE |
|---------|-----|------|
| California Housing | **~0.80** | ~0.50 |

> Gain significatif par rapport à la régression linéaire (R²=0.59 → 0.80) : la relation entre features et prix immobilier est **non-linéaire**.

---

## 15. Gradient Boosting Regressor

### Principe théorique

GBR minimise une fonction de perte L(y, F(x)) (ex: MSE pour régression) en construisant des arbres séquentiellement sur les **résidus négatifs** (pseudo-résidus) :

```
r_im = -[∂L(yi, F(x_i)) / ∂F(x_i)]_{F=F_{m-1}}
```

Pour la MSE : r_im = yi − F_{m-1}(xi) (résidus classiques)

**HistGradientBoosting :** (scikit-learn 0.21+) Implémentation très rapide par histogrammes — recommandée pour grands datasets.

### Code commenté

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300,       # 300 arbres
    learning_rate=0.1,      # eta faible -> meilleure generalisation
    max_depth=4,
    subsample=0.8,          # stochastic GBM : reduit le sur-apprentissage
    random_state=42
)
gbr.fit(X_cal_tr, y_cal_tr)
pred = gbr.predict(X_cal_te)

# Courbe de convergence : R2 apres chaque arbre
staged_r2 = [r2_score(y_cal_te, p) for p in gbr.staged_predict(X_cal_te)]
# -> determiner le nombre optimal d'arbres (early stopping manuel)
# Alternative : HistGradientBoostingRegressor(early_stopping=True)
```

### Résultats attendus

| Dataset | R² | RMSE |
|---------|-----|------|
| California Housing | **~0.83** | ~0.46 |

> Légèrement meilleur que Random Forest; souvent le **meilleur algorithme** pour la régression sur données tabulaires.

---

## 16. Support Vector Regression (SVR)

### Principe théorique

SVR cherche un tube d'ε-insensibilité autour des données : les points dans le tube ne sont pas pénalisés, ceux en dehors le sont.

**Fonction de perte ε-insensible :**

```
L_ε(y, ŷ) = max(0, |y - ŷ| - ε)
```

**Intuition :** au lieu de minimiser directement l'erreur, SVR recherche une fonction "plate" qui s'écarte d'au plus ε des vraies valeurs.

| Paramètre | Rôle |
|----------|------|
| `C` | Pénalité des exemples en dehors du tube |
| `epsilon` | Largeur du tube d'insensibilité |
| `kernel` | Transformation de l'espace (même que SVC) |

### Code commenté

```python
from sklearn.svm import SVR

svr = Pipeline([
    ("sc",  StandardScaler()),    # OBLIGATOIRE pour SVR
    ("svr", SVR(
        kernel="rbf",             # kernel gaussien
        C=100,                    # C grand : moins de regularisation
        gamma=0.1,                # gamma du kernel RBF
        epsilon=0.1               # largeur du tube
    ))
])
svr.fit(X_dia_tr, y_dia_tr)
pred = svr.predict(X_dia_te)

# SVR peut etre lent sur grands datasets (O(n^2) memoire)
# Pour grands datasets : utiliser LinearSVR ou GBR
```

### Résultats attendus (Diabetes)

| Kernel | R² |
|--------|-----|
| Linear | ~0.48 |
| **RBF** | **~0.50** |
| Poly | ~0.40 |

---

## 17. KNN Regressor

### Principe théorique

KNN Regressor prédit la valeur d'un point comme la **moyenne** (ou moyenne pondérée) des k plus proches voisins :

```
ŷ(x) = (1/k) × Σ_{xi ∈ kNN(x)} yi
```

Avec pondération par distance :

```
ŷ(x) = Σ (1/d(x,xi)) × yi / Σ (1/d(x,xi))
```

### Code commenté

```python
from sklearn.neighbors import KNeighborsRegressor

# Recherche du k optimal
for k in range(1, 21):
    knnr = Pipeline([
        ("sc",  StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance"))
    ])
    knnr.fit(X_cal_sub, y_cal_sub)
    r2 = r2_score(y_cal_te, knnr.predict(X_cal_te))
    # Optimal generalement entre k=5 et k=15

knnr_final = Pipeline([
    ("sc",  StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))
])
knnr_final.fit(X_cal_sub, y_cal_sub)
```

### Résultats attendus

| Dataset | k | R² |
|---------|---|-----|
| California Housing | 7 | **~0.68** |

---

## 18. MLP Regressor

### Principe théorique

Le MLP Regressor utilise la même architecture que le MLP Classifier, mais avec :

- **Couche de sortie :** un seul neurone linéaire (pas de softmax)
- **Fonction de perte :** MSE au lieu de l'entropie croisée
- **Métrique :** R², RMSE au lieu de l'accuracy

### Code commenté

```python
from sklearn.neural_network import MLPRegressor

mlp_reg = Pipeline([
    ("sc",  StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),  # architecture profonde
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,              # surveille MSE validation
        validation_fraction=0.1,
        learning_rate_init=0.001          # learning rate Adam
    ))
])
mlp_reg.fit(X_cal_tr, y_cal_tr)
pred = mlp_reg.predict(X_cal_te)

# Courbes d'entrainement
loss_curve = mlp_reg["mlp"].loss_curve_   # MSE train
val_scores  = mlp_reg["mlp"].validation_scores_  # R2 validation
```

### Résultats attendus

| Dataset | Architecture | R² |
|---------|-------------|-----|
| California Housing | (128, 64, 32) | **~0.80** |

---

## 19. Comparaisons globales

### Classification — Iris (test 20%)

| Algorithme | Accuracy | CV 5-fold |
|-----------|---------|---------|
| LogReg | ~97% | ~96% |
| Decision Tree | ~97% | ~94% |
| **Random Forest** | **~97%** | **~96%** |
| **Gradient Boosting** | **~97%** | **~97%** |
| **SVM (RBF)** | **~97%** | **~97%** |
| KNN (k=5) | ~97% | ~96% |
| Naive Bayes | ~97% | ~95% |
| AdaBoost | ~97% | ~95% |
| **MLP** | **~97%** | **~97%** |
| Extra Trees | ~97% | ~96%** |

> Sur Iris (problème facile), presque tous les algorithmes atteignent ~97%. Les différences apparaissent sur des datasets plus complexes (Digits, Breast Cancer).

### Régression — Diabetes

| Algorithme | RMSE | R² |
|-----------|------|-----|
| Linear Regression | ~55 | ~0.48 |
| Ridge | ~54 | ~0.49 |
| Lasso | ~55 | ~0.48 |
| ElasticNet | ~55 | ~0.48 |
| Decision Tree | ~60 | ~0.38 |
| **Random Forest** | **~52** | **~0.55** |
| **Gradient Boosting** | **~51** | **~0.57** |
| SVR (rbf) | ~53 | ~0.50 |
| KNN (k=7) | ~55 | ~0.47 |
| **MLP** | **~52** | **~0.54** |

---

## 20. Guide de choix algorithmique

### Pour la classification

| Situation | Algorithme recommandé |
|-----------|----------------------|
| Interprétabilité requise | **Régression Logistique** ou **Decision Tree** |
| Baseline rapide | **Régression Logistique** |
| Données de texte | **Naive Bayes** |
| Haute dimension | **SVM (RBF)** ou **MLP** |
| Dataset moyen, performance maximale | **Gradient Boosting** ou **Random Forest** |
| Dataset déséquilibré | **AdaBoost** ou **RF avec class_weight** |
| Peu de données | **KNN** ou **Naive Bayes** |
| Grand dataset | **MLP** ou **HistGradientBoosting** |

### Pour la régression

| Situation | Algorithme recommandé |
|-----------|----------------------|
| Relation linéaire | **Régression Linéaire (OLS)** |
| Nombreuses features inutiles | **Lasso** (feature selection) |
| Features corrélées | **Ridge** ou **ElasticNet** |
| Relation non-linéaire | **Random Forest** ou **GB** |
| Performance maximale | **Gradient Boosting** |
| Grand dataset (>100k) | **HistGradientBoosting** ou **MLP** |
| Données séquentielles/temporelles | **MLP** (ou LSTM) |

### Arbre de décision rapide

```
Données étiquetées disponibles ?
├── OUI → Supervisé
│   ├── Cible discrète → CLASSIFICATION
│   │   ├── Linéairement séparable ? → Logistic Regression / SVM Linear
│   │   ├── Interprétable ? → Decision Tree
│   │   ├── Performance max. ? → GBM + RF
│   │   └── Grande dim. ? → SVM RBF / MLP
│   └── Cible continue → REGRESSION
│       ├── Linéaire ? → OLS / Ridge / Lasso
│       ├── Non-linéaire ? → Random Forest / GBR
│       └── Très large ? → HistGBR / MLP
└── NON → Non supervisé → Clustering
```

---

## Dépendances

```bash
# Librairies de base (toutes incluses dans scikit-learn)
pip install scikit-learn scipy numpy matplotlib

# Optionnel : implementation rapide de Gradient Boosting
pip install xgboost lightgbm
```

---

## Exécution

```bash
python prediction_examples.py
```

Les graphiques apparaissent séquentiellement par algorithme. Fermer chaque fenêtre pour passer au suivant.

---

*Généré automatiquement d'après `prediction_examples.py` — Mars 2026*
