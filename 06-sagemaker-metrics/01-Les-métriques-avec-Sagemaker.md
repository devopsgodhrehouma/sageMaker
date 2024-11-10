# Lab 3.6 - Guide Pédagogique

Dans ce guide, nous allons évaluer un modèle de machine learning que nous avons entraîné dans les modules précédents. Nous calculerons des métriques basées sur les résultats des données de test pour comprendre la performance du modèle.

---

## Table des Matières

1. [Introduction](#introduction)
2. [Étape 0 : Configuration du Lab](#étape-0--configuration-du-lab)
    - [0.1. Importer les Bibliothèques et Charger les Données](#01-importer-les-bibliothèques-et-charger-les-données)
    - [0.2. Préparer et Entraîner le Modèle](#02-préparer-et-entraîner-le-modèle)
3. [Étape 1 : Exploration des Résultats](#étape-1--exploration-des-résultats)
4. [Étape 2 : Création d'une Matrice de Confusion](#étape-2--création-dune-matrice-de-confusion)
5. [Étape 3 : Calcul des Statistiques de Performance](#étape-3--calcul-des-statistiques-de-performance)
    - [3.1. Sensibilité (Recall)](#31-sensibilité-recall)
    - [3.2. Spécificité](#32-spécificité)
    - [3.3. Valeurs Prédictives Positive et Négative](#33-valeurs-prédictives-positive-et-négative)
    - [3.4. Taux de Faux Positifs et Faux Négatifs](#34-taux-de-faux-positifs-et-faux-négatifs)
    - [3.5. Taux de Faux Découvertes](#35-taux-de-faux-découvertes)
    - [3.6. Précision Globale](#36-précision-globale)
6. [Étape 4 : Calcul de la Courbe AUC-ROC](#étape-4--calcul-de-la-courbe-auc-roc)
7. [Conclusion](#conclusion)

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Introduction

Vous travaillez pour un prestataire de soins de santé et souhaitez améliorer la détection des anomalies chez les patients orthopédiques. Vous avez accès à un jeu de données contenant six caractéristiques biomécaniques et une étiquette indiquant si un patient est *normal* ou *anormal*. Vous utiliserez ce jeu de données pour entraîner un modèle de machine learning capable de prédire si un patient présente une anomalie.

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Étape 0 : Configuration du Lab

Avant de commencer l'évaluation du modèle, nous devons importer les données et entraîner le modèle pour qu'il soit prêt à être utilisé.

### 0.1. Importer les Bibliothèques et Charger les Données

```python
import warnings, requests, zipfile, io
warnings.simplefilter('ignore')
import pandas as pd
from scipy.io import arff

import os
import boto3
import sagemaker
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sagemaker.image_uris import retrieve
from sklearn.model_selection import train_test_split
```

**Explication :**

- Nous importons les bibliothèques nécessaires pour le traitement des données, l'entraînement du modèle et l'évaluation.
- Nous désactivons les avertissements pour une meilleure lisibilité.

[Retour en Haut](#lab-36---guide-pédagogique)

---

### 0.2. Préparer et Entraîner le Modèle

Nous téléchargeons le jeu de données, le préparons et entraînons un modèle XGBoost.

```python
# Télécharger et extraire les données
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Charger le fichier .arff dans un DataFrame pandas
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])

# Convertir les étiquettes de classe en valeurs numériques
class_mapper = {b'Abnormal':1, b'Normal':0}
df['class'] = df['class'].replace(class_mapper)

# Placer la colonne 'class' en première position
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# Diviser le dataset en ensembles d'entraînement, de test et de validation
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])

# Configurer le modèle XGBoost
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')
hyperparams = {"num_round": "42", "eval_metric": "auc", "objective": "binary:logistic"}
s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=s3_output_location,
    hyperparameters=hyperparams,
    sagemaker_session=sagemaker.Session()
)

# Préparer les canaux de données
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix),
    content_type='text/csv'
)
validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix),
    content_type='text/csv'
)
data_channels = {'train': train_channel, 'validation': validate_channel}

# Entraîner le modèle
xgb_model.fit(inputs=data_channels, logs=False)

# Préparer les données pour le Batch Transform
batch_X = test.iloc[:, 1:]
batch_X_file = 'batch-in.csv'
upload_s3_csv(batch_X_file, 'batch-in', batch_X)

# Configurer le Batch Transform
batch_output = "s3://{}/{}/batch-out/".format(bucket, prefix)
batch_input = "s3://{}/{}/batch-in/{}".format(bucket, prefix, batch_X_file)
xgb_transformer = xgb_model.transformer(
    instance_count=1,
    instance_type='ml.m4.xlarge',
    strategy='MultiRecord',
    assemble_with='Line',
    output_path=batch_output
)

# Lancer le Batch Transform
xgb_transformer.transform(
    data=batch_input,
    data_type='S3Prefix',
    content_type='text/csv',
    split_type='Line'
)
xgb_transformer.wait()

# Récupérer les prédictions
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key="{}/batch-out/{}".format(prefix, 'batch-in.csv.out'))
target_predicted = pd.read_csv(io.BytesIO(obj['Body'].read()), names=['class'])
```

**Explication :**

- Nous téléchargeons le jeu de données et le préparons pour l'entraînement.
- Nous configurons et entraînons un modèle XGBoost en utilisant Amazon SageMaker.
- Nous utilisons le Batch Transform pour obtenir les prédictions sur l'ensemble de test.

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Étape 1 : Exploration des Résultats

Nous commençons par explorer les résultats bruts du modèle.

```python
def binary_convert(x):
    threshold = 0.3  # Seuil pour convertir les probabilités en classes
    return 1 if x > threshold else 0

target_predicted_binary = target_predicted['class'].apply(binary_convert)
print(target_predicted_binary.head(5))
test.head(5)
```

**Explication :**

- Nous définissons une fonction pour convertir les probabilités prédites en classes binaires en utilisant un seuil.
- Nous appliquons cette fonction aux prédictions pour obtenir les classes prédites.
- Nous affichons les premières prédictions et comparons avec les valeurs réelles.

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Étape 2 : Création d'une Matrice de Confusion

Une matrice de confusion nous permet de visualiser les performances de notre modèle en comparant les prédictions avec les valeurs réelles.

```python
from sklearn.metrics import confusion_matrix

test_labels = test.iloc[:, 0]  # Étiquettes réelles
matrix = confusion_matrix(test_labels, target_predicted_binary)
df_confusion = pd.DataFrame(matrix, index=['Normal', 'Abnormal'], columns=['Normal', 'Abnormal'])
df_confusion
```

**Explication :**

- Nous extrayons les étiquettes réelles de l'ensemble de test.
- Nous utilisons `confusion_matrix` pour calculer la matrice de confusion.
- Nous créons un DataFrame pour mieux visualiser les résultats.

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Étape 3 : Calcul des Statistiques de Performance

À partir de la matrice de confusion, nous pouvons calculer plusieurs métriques pour évaluer la performance du modèle.

### 3.1. Sensibilité (Recall)

La **sensibilité** (ou Taux de Vrais Positifs - TPR) mesure la capacité du modèle à identifier correctement les échantillons positifs.

```python
TN, FP, FN, TP = confusion_matrix(test_labels, target_predicted_binary).ravel()

# Sensibilité (Recall)
Sensitivity = float(TP) / (TP + FN) * 100
print(f"Sensibilité (Recall): {Sensitivity:.2f}%")
```

**Explication :**

- **TP** (Vrais Positifs) : Nombre de cas positifs correctement prédits.
- **FN** (Faux Négatifs) : Nombre de cas positifs incorrectement prédits comme négatifs.
- La sensibilité est le ratio TP / (TP + FN).

[Retour en Haut](#lab-36---guide-pédagogique)

---

### 3.2. Spécificité

La **spécificité** (ou Taux de Vrais Négatifs - TNR) mesure la capacité du modèle à identifier correctement les échantillons négatifs.

```python
# Spécificité
Specificity = float(TN) / (TN + FP) * 100
print(f"Spécificité: {Specificity:.2f}%")
```

**Explication :**

- **TN** (Vrais Négatifs) : Nombre de cas négatifs correctement prédits.
- **FP** (Faux Positifs) : Nombre de cas négatifs incorrectement prédits comme positifs.
- La spécificité est le ratio TN / (TN + FP).

[Retour en Haut](#lab-36---guide-pédagogique)

---

### 3.3. Valeurs Prédictives Positive et Négative

**Valeur Prédictive Positive (Précision)** : Probabilité que les cas prédits positifs soient réellement positifs.

```python
# Précision (Valeur Prédictive Positive)
Precision = float(TP) / (TP + FP) * 100
print(f"Précision: {Precision:.2f}%")
```

**Valeur Prédictive Négative** : Probabilité que les cas prédits négatifs soient réellement négatifs.

```python
# Valeur Prédictive Négative
NPV = float(TN) / (TN + FN) * 100
print(f"Valeur Prédictive Négative: {NPV:.2f}%")
```

[Retour en Haut](#lab-36---guide-pédagogique)

---

### 3.4. Taux de Faux Positifs et Faux Négatifs

**Taux de Faux Positifs (FPR)** : Probabilité que le modèle prédit positifs alors que le cas est négatif.

```python
# Taux de Faux Positifs
FPR = float(FP) / (FP + TN) * 100
print(f"Taux de Faux Positifs (FPR): {FPR:.2f}%")
```

**Taux de Faux Négatifs (FNR)** : Probabilité que le modèle prédit négatifs alors que le cas est positif.

```python
# Taux de Faux Négatifs
FNR = float(FN) / (TP + FN) * 100
print(f"Taux de Faux Négatifs (FNR): {FNR:.2f}%")
```

[Retour en Haut](#lab-36---guide-pédagogique)

---

### 3.5. Taux de Faux Découvertes

Le **Taux de Faux Découvertes (FDR)** est la probabilité que les prédictions positives soient incorrectes.

```python
# Taux de Faux Découvertes
FDR = float(FP) / (TP + FP) * 100
print(f"Taux de Faux Découvertes (FDR): {FDR:.2f}%")
```

[Retour en Haut](#lab-36---guide-pédagogique)

---

### 3.6. Précision Globale

La **précision globale** mesure le pourcentage total de prédictions correctes.

```python
# Précision Globale
ACC = float(TP + TN) / (TP + FP + FN + TN) * 100
print(f"Précision Globale: {ACC:.2f}%")
```

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Étape 4 : Calcul de la Courbe AUC-ROC

La **courbe ROC** (Receiver Operating Characteristic) est une représentation graphique de la performance d'un modèle de classification binaire.

```python
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Calcul de l'AUC
auc_score = roc_auc_score(test_labels, target_predicted)
print(f"AUC Score: {auc_score:.2f}")

# Calcul des valeurs pour la courbe ROC
fpr, tpr, thresholds = roc_curve(test_labels, target_predicted)

# Tracé de la courbe ROC
plt.figure()
plt.plot(fpr, tpr, label=f'Courbe ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonale pointillée
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()
```

**Explication :**

- **AUC** (Area Under the Curve) quantifie la capacité du modèle à distinguer entre les classes.
- Une AUC proche de 1 indique une excellente performance.

[Retour en Haut](#lab-36---guide-pédagogique)

---

## Conclusion

Nous avons évalué notre modèle de classification binaire en utilisant diverses métriques dérivées de la matrice de confusion. Ces métriques nous aident à comprendre les forces et les faiblesses du modèle.

**Points Clés :**

- **Sensibilité** élevée indique que le modèle détecte bien les cas positifs.
- **Spécificité** plus faible peut signifier que le modèle a du mal à identifier correctement les cas négatifs.
- **AUC-ROC** nous donne une vision globale de la capacité du modèle à distinguer entre les classes.

**Prochaines Étapes :**

- **Ajuster le Seuil** : En modifiant le seuil de conversion des probabilités en classes (par exemple, tester 0.25 ou 0.75), nous pouvons observer comment les métriques changent.
- **Améliorer le Modèle** : En ajustant les hyperparamètres ou en utilisant d'autres algorithmes.
- **Évaluer l'Impact** : Comprendre les implications des faux positifs et faux négatifs dans le contexte médical pour améliorer la prise de décision.

**Félicitations !** Vous avez terminé l'évaluation de votre modèle et êtes maintenant prêt à l'améliorer davantage.

[Retour en Haut](#lab-36---guide-pédagogique)

# Remerciements

Merci d'avoir suivi ce guide pédagogique. N'hésitez pas à revisiter les sections pour approfondir votre compréhension.

[Retour en Haut](#lab-36---guide-pédagogique)
