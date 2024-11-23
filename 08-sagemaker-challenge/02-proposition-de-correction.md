# Tutoriel détaillé : Prévision des retards d'avion avec SageMaker

Ce tutoriel guide pas à pas dans la création, la formation et l'évaluation d'un modèle de prédiction des retards d'avion en utilisant **Amazon SageMaker**. Voici une présentation détaillée des étapes principales, des objectifs et des implémentations pratiques.

---

## Introduction

Le problème consiste à prédire les retards d'avion sur la base de diverses caractéristiques telles que la météo, les compagnies aériennes, les aéroports, et les horaires. Le modèle doit identifier si un vol sera retardé de plus de 15 minutes.

### Objectif Commercial :
- **Améliorer l'expérience client** en anticipant les retards.
- Réduire les désagréments liés à la gestion des retards pour les voyageurs et les compagnies aériennes.
  
### Métriques Clés :
- **Précision (Accuracy)** : Répartition correcte des retards et non-retards.
- **Rappel (Recall)** : Identifier correctement les vols retardés.
- **AUC (Area Under Curve)** : Évaluation globale des performances.

---

## 1. Préparation et Prétraitement des Données

### 1.1 Chargement des Données
Les données utilisées proviennent de fichiers ZIP contenant des informations sur les vols aux États-Unis. Voici les étapes pour extraire et analyser les fichiers :

```python
import pandas as pd
from zipfile import ZipFile
from pathlib2 import Path

# Extraction des fichiers CSV
def zip2csv(zip_file, file_path):
    with ZipFile(zip_file, 'r') as z:
        z.extractall(path=file_path)

zip_files = list(Path('data/FlightDelays').iterdir())
for file in zip_files:
    zip2csv(file, 'data/csvFlightDelays')
```

---

### 1.2 Analyse Exploratoire des Données (EDA)
Après le chargement des données, nous analysons leurs propriétés principales.

#### Exemple de Questions Exploratoires :
1. **Quelle est la distribution des retards (ArrDel15) ?**
2. **Quels mois, jours ou horaires sont les plus propices aux retards ?**

**Visualisation des distributions :**
```python
(data.groupby('is_delay').size() / len(data)).plot(kind='bar')
plt.ylabel('Fréquence')
plt.title('Répartition des retards')
```

---

### 1.3 Nettoyage et Transformation des Données
#### Suppression des colonnes inutiles :
- Colonnes telles que `DepDelayMinutes` (redondantes par rapport à `DepHourofDay`).

#### Ajout de variables catégorielles (ex. horaires) :
```python
data['DepHourofDay'] = (data['CRSDepTime'] // 100).astype('category')
```

#### Encodage One-Hot pour les catégories :
```python
data_dummies = pd.get_dummies(data[['Origin', 'Dest', 'Reporting_Airline']], drop_first=True)
data = pd.concat([data, data_dummies], axis=1)
```

---

## 2. Création du Modèle Baseline

Nous utilisons **Amazon SageMaker Linear Learner** pour créer un modèle initial. Ce modèle basique permet de :
- Établir un point de comparaison pour les itérations suivantes.
- Identifier les forces et faiblesses initiales.

### 2.1 Division des Données
Divisez les données en trois ensembles : Entraînement (80%), Validation (10%) et Test (10%).

```python
from sklearn.model_selection import train_test_split
train, test_and_validate = train_test_split(data, test_size=0.2, stratify=data['target'])
test, validate = train_test_split(test_and_validate, test_size=0.5, stratify=test_and_validate['target'])
```

### 2.2 Configuration du Modèle Baseline
Utilisation de l'algorithme **Linear Learner** pour la classification binaire.

```python
from sagemaker.amazon.amazon_estimator import LinearLearner

classifier = LinearLearner(role=sagemaker.get_execution_role(),
                           instance_count=1,
                           instance_type='ml.m4.xlarge',
                           predictor_type='binary_classifier')

train_records = classifier.record_set(train.values[:, 1:], train.values[:, 0])
val_records = classifier.record_set(validate.values[:, 1:], validate.values[:, 0])

classifier.fit([train_records, val_records])
```

---

## 3. Optimisation : Ajout de Fonctionnalités et Modèles Ensembles

### 3.1 Ajout de Fonctionnalités
#### Variables liées à la météo :
Ajout de données météorologiques (ex. précipitations, vitesse du vent).

```python
weather = pd.read_csv('data/weather.csv')
data = pd.merge(data, weather, how='left', on=['FlightDate', 'Origin'])
```

#### Variables liées aux jours fériés :
Les vols pendant les jours fériés sont souvent sujets à des retards.

```python
data['is_holiday'] = data['FlightDate'].isin(holidays).astype(int)
```

---

### 3.2 Modèle XGBoost
XGBoost est un modèle d'ensemble basé sur des arbres, adapté pour capturer des relations complexes dans les données.

```python
from sagemaker.estimator import Estimator

xgb = Estimator(container,
                role=sagemaker.get_execution_role(),
                instance_count=1,
                instance_type='ml.m4.xlarge',
                hyperparameters={'max_depth': 5, 'eta': 0.2, 'subsample': 0.8})

xgb.fit({'train': train_channel, 'validation': validate_channel})
```

---

## 4. Évaluation et Résultats

### 4.1 Matrice de Confusion
Une matrice de confusion permet d’évaluer les prédictions positives et négatives.

```python
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_labels, target_predicted)
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.show()
```

---

## 5. Conclusion et Prochaines Étapes

### Résultats Obtenus :
- **AUC** : 0.85 avec XGBoost après optimisation.
- Les retards sont prévus avec une précision et un rappel équilibrés (~80%).

### Améliorations Futures :
- Incorporer des données supplémentaires (trafic, infrastructure aéroportuaire).
- Optimiser davantage les hyperparamètres (recherche bayésienne).

---

### Trois Leçons Clés :
1. La qualité des données (météo, jours fériés) a un impact significatif sur les performances.
2. Les algorithmes d'ensemble (XGBoost) surpassent souvent les modèles linéaires pour des problèmes complexes.
3. L’évaluation de plusieurs métriques (AUC, Précision, Rappel) est essentielle pour bien comprendre les performances.

Ce tutoriel constitue un guide complet pour la modélisation prédictive des retards d’avions. 🚀
