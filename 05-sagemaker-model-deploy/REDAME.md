
# Guide Pédagogique : Étapes Clés du Déploiement d'un Modèle de Machine Learning avec SageMaker

Ce guide vous accompagne à travers les étapes essentielles pour déployer un modèle de machine learning en utilisant Amazon SageMaker. Chaque section est conçue pour vous aider à comprendre le processus de manière claire et pédagogique.

---

## Table des Matières

1. [Introduction](#introduction)
2. [Étape 1 : Configuration de l'Environnement et Importation des Données](#étape-1--configuration-de-lenvironnement-et-importation-des-données)
    - [1.1. Importer les Bibliothèques Nécessaires](#11-importer-les-bibliothèques-nécessaires)
    - [1.2. Télécharger et Charger le Jeu de Données](#12-télécharger-et-charger-le-jeu-de-données)
    - [1.3. Préparer les Données pour l'Entraînement](#13-préparer-les-données-pour-lentraînement)
    - [1.4. Diviser le Dataset en Ensembles d'Entraînement, de Test et de Validation](#14-diviser-le-dataset-en-ensembles-dentraînement-de-test-et-de-validation)
    - [1.5. Téléverser les Données sur Amazon S3](#15-téléverser-les-données-sur-amazon-s3)
3. [Étape 2 : Configuration et Entraînement du Modèle XGBoost](#étape-2--configuration-et-entraînement-du-modèle-xgboost)
    - [2.1. Récupérer l'URI de l'Image Docker pour XGBoost](#21-récupérer-luri-de-limage-docker-pour-xgboost)
    - [2.2. Définir les Hyperparamètres du Modèle](#22-définir-les-hyperparamètres-du-modèle)
    - [2.3. Créer l'Estimateur et Spécifier les Canaux de Données](#23-créer-lestimateur-et-spécifier-les-canaux-de-données)
    - [2.4. Entraîner le Modèle](#24-entraîner-le-modèle)
4. [Étape 3 : Déployer le Modèle pour les Prédictions en Temps Réel](#étape-3--déployer-le-modèle-pour-les-prédictions-en-temps-réel)
    - [3.1. Déployer le Modèle sur un Endpoint SageMaker](#31-déployer-le-modèle-sur-un-endpoint-sagemaker)
5. [Étape 4 : Effectuer des Prédictions Individuelles](#étape-4--effectuer-des-prédictions-individuelles)
    - [4.1. Préparer une Ligne de Données pour la Prédiction](#41-préparer-une-ligne-de-données-pour-la-prédiction)
    - [4.2. Convertir la Ligne en Format CSV](#42-convertir-la-ligne-en-format-csv)
    - [4.3. Envoyer la Requête de Prédiction au Endpoint](#43-envoyer-la-requête-de-prédiction-au-endpoint)
    - [4.4. Interpréter la Prédiction](#44-interpréter-la-prédiction)
6. [Étape 5 : Supprimer le Endpoint pour Libérer les Ressources](#étape-5--supprimer-le-endpoint-pour-libérer-les-ressources)
7. [Étape 6 : Effectuer une Transformation par Lot (Batch Transform)](#étape-6--effectuer-une-transformation-par-lot-batch-transform)
    - [6.1. Préparer les Données pour le Batch Transform](#61-préparer-les-données-pour-le-batch-transform)
    - [6.2. Téléverser les Données sur S3](#62-téléverser-les-données-sur-s3)
    - [6.3. Configurer le Batch Transform](#63-configurer-le-batch-transform)
    - [6.4. Exécuter le Batch Transform](#64-exécuter-le-batch-transform)
    - [6.5. Télécharger et Analyser les Résultats](#65-télécharger-et-analyser-les-résultats)
8. [Conclusion](#conclusion)

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Introduction

Dans ce guide, nous allons explorer comment déployer un modèle de machine learning en utilisant Amazon SageMaker. Nous travaillerons avec un jeu de données biomécaniques pour prédire si un patient présente une anomalie orthopédique.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Étape 1 : Configuration de l'Environnement et Importation des Données

### 1.1. Importer les Bibliothèques Nécessaires

```python
import warnings
import requests
import zipfile
import io
import pandas as pd
from scipy.io import arff
import os
import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sklearn.model_selection import train_test_split
```

**Explication :**

- **warnings** : Pour gérer les avertissements.
- **requests** : Pour effectuer des requêtes HTTP.
- **zipfile** et **io** : Pour manipuler des fichiers zip et les flux de données.
- **pandas** : Pour la manipulation des données.
- **scipy.io.arff** : Pour lire les fichiers .arff.
- **boto3** : Pour interagir avec les services AWS.
- **sagemaker** : SDK pour utiliser SageMaker.
- **train_test_split** : Pour diviser le dataset.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 1.2. Télécharger et Charger le Jeu de Données

```python
# Télécharger le fichier zip contenant le dataset
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Charger le fichier .arff dans un DataFrame pandas
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])
```

**Explication :**

- Nous téléchargeons le dataset et l'extrayons.
- Le fichier `.arff` est chargé dans un DataFrame pandas.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 1.3. Préparer les Données pour l'Entraînement

```python
# Convertir les étiquettes de classe en valeurs numériques
class_mapper = {b'Abnormal':1, b'Normal':0}
df['class'] = df['class'].replace(class_mapper)

# Placer la colonne 'class' en première position
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
```

**Explication :**

- Les classes sont mappées à des valeurs numériques.
- La colonne 'class' est déplacée en première position.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 1.4. Diviser le Dataset en Ensembles d'Entraînement, de Test et de Validation

```python
# Diviser le dataset
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])
```

**Explication :**

- Le dataset est divisé en ensembles d'entraînement, de test et de validation.
- La stratification assure une proportion égale des classes dans chaque ensemble.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 1.5. Téléverser les Données sur Amazon S3

```python
# Fonction pour téléverser un DataFrame sur S3
def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())

# Téléverser les ensembles de données
upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)
```

**Explication :**

- Les données sont converties en CSV et téléversées sur S3.
- Chaque ensemble est stocké dans un dossier spécifique.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Étape 2 : Configuration et Entraînement du Modèle XGBoost

### 2.1. Récupérer l'URI de l'Image Docker pour XGBoost

```python
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')
```

**Explication :**

- Récupération de l'image Docker pour la version de XGBoost compatible avec SageMaker.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 2.2. Définir les Hyperparamètres du Modèle

```python
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic"
}
```

**Explication :**

- **num_round** : Nombre d'itérations d'entraînement.
- **eval_metric** : Métrique d'évaluation (AUC).
- **objective** : Objectif de l'entraînement (classification binaire).

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 2.3. Créer l'Estimateur et Spécifier les Canaux de Données

```python
# Créer l'estimateur
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=s3_output_location,
    hyperparameters=hyperparams,
    sagemaker_session=sagemaker.Session()
)

# Spécifier les canaux de données
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix),
    content_type='text/csv'
)
validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix),
    content_type='text/csv'
)
data_channels = {'train': train_channel, 'validation': validate_channel}
```

**Explication :**

- L'estimateur configure comment SageMaker entraînera le modèle.
- Les canaux de données pointent vers les emplacements S3 des données.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 2.4. Entraîner le Modèle

```python
# Entraîner le modèle
xgb_model.fit(inputs=data_channels, logs=False)
```

**Explication :**

- Le modèle est entraîné en utilisant les données fournies.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Étape 3 : Déployer le Modèle pour les Prédictions en Temps Réel

### 3.1. Déployer le Modèle sur un Endpoint SageMaker

```python
# Déployer le modèle
xgb_predictor = xgb_model.deploy(
    initial_instance_count=1,
    serializer=sagemaker.serializers.CSVSerializer(),
    instance_type='ml.m4.xlarge'
)
```

**Explication :**

- Le modèle est déployé sur un endpoint pour permettre les prédictions en temps réel.
- Le sérialiseur spécifie le format des données d'entrée (CSV).

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Étape 4 : Effectuer des Prédictions Individuelles

### 4.1. Préparer une Ligne de Données pour la Prédiction

```python
# Sélectionner une ligne sans la colonne 'class'
row = test.iloc[0:1, 1:]
```

**Explication :**

- Nous préparons une ligne de données en excluant la colonne cible.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 4.2. Convertir la Ligne en Format CSV

```python
# Convertir la ligne en CSV
batch_X_csv_buffer = io.StringIO()
row.to_csv(batch_X_csv_buffer, header=False, index=False)
test_row = batch_X_csv_buffer.getvalue()
```

**Explication :**

- La ligne est convertie en une chaîne CSV pour être compatible avec le modèle.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 4.3. Envoyer la Requête de Prédiction au Endpoint

```python
# Obtenir la prédiction
prediction = xgb_predictor.predict(test_row)
print(prediction)
```

**Explication :**

- La prédiction est obtenue en envoyant la donnée au endpoint.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 4.4. Interpréter la Prédiction

**Explication :**

- Le résultat est une probabilité indiquant la probabilité que le patient soit 'Anormal'.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Étape 5 : Supprimer le Endpoint pour Libérer les Ressources

```python
# Supprimer le endpoint
xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```

**Explication :**

- La suppression du endpoint libère les ressources et évite des coûts supplémentaires.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Étape 6 : Effectuer une Transformation par Lot (Batch Transform)

### 6.1. Préparer les Données pour le Batch Transform

```python
# Préparer les données
batch_X = test.iloc[:, 1:]
```

**Explication :**

- Nous sélectionnons toutes les caractéristiques de l'ensemble de test pour le Batch Transform.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 6.2. Téléverser les Données sur S3

```python
# Téléverser les données sur S3
batch_X_file = 'batch-in.csv'
upload_s3_csv(batch_X_file, 'batch-in', batch_X)
```

**Explication :**

- Les données sont téléversées sur S3 pour être utilisées dans le Batch Transform.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 6.3. Configurer le Batch Transform

```python
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
```

**Explication :**

- Nous configurons le Batch Transform avec les emplacements d'entrée et de sortie.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 6.4. Exécuter le Batch Transform

```python
# Lancer le Batch Transform
xgb_transformer.transform(
    data=batch_input,
    data_type='S3Prefix',
    content_type='text/csv',
    split_type='Line'
)
xgb_transformer.wait()
```

**Explication :**

- Le Batch Transform est exécuté pour générer des prédictions sur l'ensemble de données.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

### 6.5. Télécharger et Analyser les Résultats

```python
# Télécharger les résultats
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key="{}/batch-out/{}".format(prefix, 'batch-in.csv.out'))
target_predicted = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=',', names=['class'])

# Convertir les probabilités en classes binaires
def binary_convert(x):
    threshold = 0.65
    return 1 if x > threshold else 0

target_predicted['binary'] = target_predicted['class'].apply(binary_convert)
```

**Explication :**

- Les résultats sont téléchargés depuis S3 et analysés.
- Les probabilités sont converties en classes binaires pour l'évaluation.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)

---

## Conclusion

Nous avons parcouru les étapes clés pour déployer un modèle de machine learning avec SageMaker :

1. **Configuration de l'environnement et importation des données.**
2. **Entraînement du modèle XGBoost.**
3. **Déploiement pour des prédictions en temps réel.**
4. **Prédictions individuelles et interprétation.**
5. **Suppression du endpoint pour gérer les ressources.**
6. **Transformation par lot pour prédictions à grande échelle.**

**Points Clés :**

- **Préparation des données** est essentielle pour un modèle performant.
- **Gestion des ressources** aide à optimiser les coûts.
- **Évaluation des résultats** permet d'améliorer le modèle.

**Prochaines Étapes :**

- Calculer des métriques d'évaluation.
- Ajuster les hyperparamètres pour améliorer les performances.
- Explorer d'autres algorithmes ou techniques de prétraitement.

**Félicitations !** Vous avez appris à déployer et utiliser un modèle de machine learning avec SageMaker de manière pédagogique.

[Retour en Haut](#guide-pédagogique--étapes-clés-du-déploiement-dun-modèle-de-machine-learning-avec-sagemaker)
