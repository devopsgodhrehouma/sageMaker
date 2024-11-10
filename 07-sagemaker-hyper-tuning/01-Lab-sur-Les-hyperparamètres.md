# Lab 3.7 - Guide Pédagogique

Dans ce guide, nous allons créer un travail d'ajustement d'hyperparamètres pour améliorer le modèle que nous avons précédemment entraîné. Nous comparerons ensuite les métriques des deux modèles pour évaluer les améliorations.

---

## Table des Matières

1. [Introduction](#introduction)
2. [Étape 0 : Configuration du Lab](#étape-0--configuration-du-lab)
    - [0.1. Importer les Bibliothèques et Charger les Données](#01-importer-les-bibliothèques-et-charger-les-données)
    - [0.2. Préparer et Entraîner le Modèle Initial](#02-préparer-et-entraîner-le-modèle-initial)
3. [Étape 1 : Obtenir les Statistiques du Modèle Initial](#étape-1--obtenir-les-statistiques-du-modèle-initial)
4. [Étape 2 : Créer un Travail d'Ajustement d'Hyperparamètres](#étape-2--créer-un-travail-dajustement-dhyperparamètres)
5. [Étape 3 : Examiner les Résultats du Travail d'Ajustement](#étape-3--examiner-les-résultats-du-travail-dajustement)
6. [Étape 4 : Évaluer le Modèle Optimisé](#étape-4--évaluer-le-modèle-optimisé)
7. [Conclusion](#conclusion)

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Introduction

Vous travaillez pour un prestataire de soins de santé et souhaitez améliorer la détection des anomalies chez les patients orthopédiques. Vous avez accès à un jeu de données contenant six caractéristiques biomécaniques et une étiquette indiquant si un patient est *normal* ou *anormal*. Vous utiliserez ce jeu de données pour entraîner un modèle de machine learning capable de prédire si un patient présente une anomalie.

Dans ce lab, nous allons créer un travail d'ajustement d'hyperparamètres pour optimiser le modèle précédemment entraîné. Nous comparerons ensuite les performances du nouveau modèle avec celles de l'ancien.

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Étape 0 : Configuration du Lab

Avant de commencer, nous devons importer les données et entraîner un modèle de base qui servira de point de comparaison.

### 0.1. Importer les Bibliothèques et Charger les Données

```python
import warnings, requests, zipfile, io
warnings.simplefilter('ignore')
import pandas as pd
from scipy.io import arff

import os
import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

**Explication :**

- Nous importons les bibliothèques nécessaires pour le traitement des données, l'entraînement du modèle, et l'évaluation.
- Nous désactivons les avertissements pour une meilleure lisibilité.

[Retour en Haut](#lab-37---guide-pédagogique)

---

### 0.2. Préparer et Entraîner le Modèle Initial

Nous téléchargeons le jeu de données, le préparons et entraînons un modèle XGBoost de base.

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

# Définir le préfixe pour S3
prefix = 'lab3'

# Préparer les fichiers pour S3
train_file = 'vertebral_train.csv'
test_file = 'vertebral_test.csv'
validate_file = 'vertebral_validate.csv'

# Fonction pour téléverser un DataFrame sur S3
def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())

# Téléverser les ensembles de données sur S3
s3_resource = boto3.Session().resource('s3')
upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)

# Récupérer l'URI du conteneur XGBoost
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')

# Définir les hyperparamètres du modèle initial
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "silent": 1
}

# Configurer l'estimateur XGBoost
s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.2xlarge',
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
```

**Explication :**

- Nous téléchargeons et préparons les données pour l'entraînement.
- Nous configurons et entraînons un modèle XGBoost de base en utilisant Amazon SageMaker.
- Les données sont divisées en ensembles d'entraînement, de test et de validation.
- Les données sont téléversées sur Amazon S3 pour être utilisées par SageMaker.

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Étape 1 : Obtenir les Statistiques du Modèle Initial

Avant d'ajuster le modèle, nous allons évaluer ses performances actuelles pour disposer d'un point de référence.

### 1.1. Effectuer des Prédictions sur l'Ensemble de Test

```python
# Préparer les données pour le Batch Transform
batch_X = test.iloc[:, 1:]
batch_X_file = 'batch-in.csv'
upload_s3_csv(batch_X_file, 'batch-in', batch_X)

# Configurer le Batch Transform
batch_output = "s3://{}/{}/batch-out/".format(bucket, prefix)
batch_input = "s3://{}/{}/batch-in/{}".format(bucket, prefix, batch_X_file)
xgb_transformer = xgb_model.transformer(
    instance_count=1,
    instance_type='ml.m5.2xlarge',
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
xgb_transformer.wait(logs=False)
```

**Explication :**

- Nous utilisons le Batch Transform pour obtenir les prédictions du modèle sur l'ensemble de test.

### 1.2. Récupérer et Préparer les Prédictions

```python
# Récupérer les prédictions depuis S3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key="{}/batch-out/{}".format(prefix, 'batch-in.csv.out'))
target_predicted = pd.read_csv(io.BytesIO(obj['Body'].read()), names=['class'])

# Convertir les probabilités en classes binaires
def binary_convert(x):
    threshold = 0.5
    return 1 if x > threshold else 0

target_predicted_binary = target_predicted['class'].apply(binary_convert)
test_labels = test.iloc[:, 0]  # Étiquettes réelles
```

**Explication :**

- Nous téléchargeons les prédictions du modèle et les convertissons en classes binaires en utilisant un seuil de 0.5.
- Nous extrayons les étiquettes réelles de l'ensemble de test pour la comparaison.

### 1.3. Évaluer le Modèle Initial

#### Matrice de Confusion

```python
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(test_labels, target_predicted):
    matrix = confusion_matrix(test_labels, target_predicted)
    df_confusion = pd.DataFrame(matrix)
    colormap = sns.color_palette("BrBG", 10)
    sns.heatmap(df_confusion, annot=True, fmt='d', cbar=None, cmap=colormap)
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    plt.ylabel("Classe Réelle")
    plt.xlabel("Classe Prédite")
    plt.show()

plot_confusion_matrix(test_labels, target_predicted_binary)
```

**Explication :**

- Nous traçons la matrice de confusion pour visualiser les performances du modèle.

#### Courbe ROC et AUC

```python
def plot_roc(test_labels, target_predicted_binary):
    fpr, tpr, thresholds = roc_curve(test_labels, target_predicted_binary)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonale pointillée
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Afficher les métriques
    print(f"AUC du Modèle Initial: {roc_auc_score(test_labels, target_predicted_binary):.2f}")

plot_roc(test_labels, target_predicted_binary)
```

**Explication :**

- Nous traçons la courbe ROC et calculons l'AUC pour le modèle initial.

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Étape 2 : Créer un Travail d'Ajustement d'Hyperparamètres

Nous allons maintenant configurer et lancer un travail d'ajustement d'hyperparamètres pour optimiser notre modèle.

### 2.1. Configurer le Tuner

```python
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

# Créer un nouvel estimateur pour le tuning
xgb = sagemaker.estimator.Estimator(
    container,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    sagemaker_session=sagemaker.Session()
)

# Définir les hyperparamètres de base
xgb.set_hyperparameters(
    eval_metric='error@0.40',
    objective='binary:logistic',
    num_round=42
)

# Définir les hyperparamètres à ajuster et leurs plages
hyperparameter_ranges = {
    'alpha': ContinuousParameter(0, 100),
    'min_child_weight': ContinuousParameter(1, 5),
    'subsample': ContinuousParameter(0.5, 1),
    'eta': ContinuousParameter(0.1, 0.3),
    'num_round': IntegerParameter(1, 50)
}

# Spécifier la métrique objective
objective_metric_name = 'validation:error'
objective_type = 'Minimize'

# Configurer le tuner
tuner = HyperparameterTuner(
    xgb,
    objective_metric_name,
    hyperparameter_ranges,
    max_jobs=10,  # Nombre total de jobs
    max_parallel_jobs=1,  # Jobs parallèles
    objective_type=objective_type,
    early_stopping_type='Auto'
)

# Lancer le travail d'ajustement
tuner.fit(inputs=data_channels, include_cls_metadata=False)
```

**Explication :**

- Nous créons un nouvel estimateur XGBoost pour l'ajustement.
- Nous définissons les hyperparamètres que nous souhaitons ajuster, ainsi que leurs plages de valeurs.
- Nous spécifions la métrique objective que le tuner doit utiliser pour évaluer les modèles.
- Nous configurons le tuner avec le nombre de jobs total et le nombre de jobs parallèles.
- Nous lançons le travail d'ajustement.

**Remarque :** Le travail d'ajustement peut prendre un certain temps (environ 45 minutes). Vous pouvez surveiller le statut du job dans la console AWS.

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Étape 3 : Examiner les Résultats du Travail d'Ajustement

Une fois le travail d'ajustement terminé, nous pouvons examiner les résultats pour identifier le meilleur modèle.

### 3.1. Vérifier le Statut du Travail

```python
# Vérifier que le travail est terminé
status = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner.latest_tuning_job.name)['HyperParameterTuningJobStatus']
print(f"Statut du travail d'ajustement : {status}")
```

### 3.2. Récupérer les Meilleurs Hyperparamètres

```python
from sagemaker.analytics import HyperparameterTuningJobAnalytics

# Obtenir les analyses du travail d'ajustement
tuner_analytics = HyperparameterTuningJobAnalytics(tuner.latest_tuning_job.name, sagemaker_session=sagemaker.Session())
df_tuning_job_analytics = tuner_analytics.dataframe()

# Trier les résultats par valeur de métrique
df_tuning_job_analytics.sort_values(
    by=['FinalObjectiveValue'],
    inplace=True,
    ascending=True  # Nous minimisons l'erreur
)

# Afficher les meilleurs résultats
df_tuning_job_analytics.head()
```

**Explication :**

- Nous récupérons les résultats du travail d'ajustement et les affichons dans un DataFrame pandas.
- Nous trions les modèles en fonction de la valeur finale de la métrique objective (erreur).
- Nous identifions le meilleur modèle en fonction de la métrique.

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Étape 4 : Évaluer le Modèle Optimisé

### 4.1. Attacher le Meilleur Modèle

```python
from sagemaker.estimator import Estimator

# Attacher le meilleur job d'entraînement
attached_tuner = HyperparameterTuner.attach(tuner.latest_tuning_job.name, sagemaker_session=sagemaker.Session())
best_training_job = attached_tuner.best_training_job()

# Créer le modèle à partir du meilleur job
algo_estimator = Estimator.attach(best_training_job)
best_algo_model = algo_estimator.create_model(env={'SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT': "text/csv"})
```

**Explication :**

- Nous attachons le meilleur job d'entraînement pour pouvoir l'utiliser.
- Nous créons un modèle à partir du meilleur job.

### 4.2. Effectuer des Prédictions avec le Modèle Optimisé

```python
# Utiliser le Batch Transform pour obtenir les prédictions
batch_output = "s3://{}/{}/batch-out/".format(bucket, prefix)
batch_input = "s3://{}/{}/batch-in/{}".format(bucket, prefix, batch_X_file)

xgb_transformer = best_algo_model.transformer(
    instance_count=1,
    instance_type='ml.m4.xlarge',
    strategy='MultiRecord',
    assemble_with='Line',
    output_path=batch_output
)

xgb_transformer.transform(
    data=batch_input,
    data_type='S3Prefix',
    content_type='text/csv',
    split_type='Line'
)
xgb_transformer.wait(logs=False)
```

### 4.3. Récupérer et Préparer les Prédictions du Modèle Optimisé

```python
# Récupérer les prédictions
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key="{}/batch-out/{}".format(prefix, 'batch-in.csv.out'))
best_target_predicted = pd.read_csv(io.BytesIO(obj['Body'].read()), names=['class'])

# Convertir les probabilités en classes binaires
best_target_predicted_binary = best_target_predicted['class'].apply(binary_convert)
```

### 4.4. Évaluer le Modèle Optimisé

#### Matrice de Confusion

```python
plot_confusion_matrix(test_labels, best_target_predicted_binary)
```

#### Courbe ROC et AUC

```python
plot_roc(test_labels, best_target_predicted_binary)
```

**Explication :**

- Nous évaluons le modèle optimisé de la même manière que le modèle initial.
- Nous comparons les métriques pour voir si le modèle optimisé est meilleur.

### 4.5. Comparaison des Modèles

**Questions :**

- Comment les résultats du modèle optimisé diffèrent-ils de ceux du modèle initial ?
- Les métriques se sont-elles améliorées ?
- Le modèle optimisé est-il meilleur que le modèle initial ?

**Réponse :**

- En comparant les métriques (AUC, précision, rappel, etc.), nous pouvons déterminer si le modèle optimisé offre de meilleures performances.
- Si les métriques du modèle optimisé sont supérieures, cela signifie que l'ajustement des hyperparamètres a été bénéfique.

[Retour en Haut](#lab-37---guide-pédagogique)

---

## Conclusion

Dans ce lab, nous avons :

- Créé un travail d'ajustement d'hyperparamètres pour optimiser notre modèle de classification binaire.
- Examiné les résultats du travail d'ajustement pour identifier le meilleur ensemble d'hyperparamètres.
- Évalué le modèle optimisé et comparé ses performances avec celles du modèle initial.

**Points Clés :**

- L'ajustement d'hyperparamètres peut améliorer significativement les performances d'un modèle.
- L'utilisation d'un travail d'ajustement permet d'explorer efficacement différentes combinaisons d'hyperparamètres.
- Il est important de comparer les métriques avant et après l'ajustement pour évaluer l'amélioration.

**Prochaines Étapes :**

- Augmenter le nombre de jobs et étendre les plages des hyperparamètres pour explorer davantage de combinaisons.
- Expérimenter avec d'autres algorithmes ou techniques de prétraitement des données.
- Déployer le modèle optimisé en production si les performances sont satisfaisantes.

**Félicitations !** Vous avez terminé ce lab et avez appris à optimiser un modèle de machine learning en ajustant ses hyperparamètres.

[Retour en Haut](#lab-37---guide-pédagogique)
