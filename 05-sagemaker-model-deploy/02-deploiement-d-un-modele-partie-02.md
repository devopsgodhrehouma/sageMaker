# Explication détaillée du code par étapes

*Dans ce guide, nous allons explorer comment déployer un modèle de machine learning entraîné, effectuer des prédictions individuelles et par lot, et finalement évaluer les résultats. Nous utiliserons Amazon SageMaker pour faciliter ce processus. Chaque étape sera accompagnée d'explications détaillées pour vous aider à comprendre ce que nous faisons et pourquoi nous le faisons.*

## Table des matières

1. [Introduction](#introduction)
2. [Configuration du lab](#configuration-du-lab)
   - [Importation des bibliothèques et des données](#importation-des-bibliothèques-et-des-données)
   - [Préparation des données](#préparation-des-données)
   - [Téléchargement des données sur S3](#téléchargement-des-données-sur-s3)
   - [Configuration et entraînement du modèle XGBoost](#configuration-et-entraînement-du-modèle-xgboost)
3. [Étape 1 : Hébergement du modèle](#étape-1--hébergement-du-modèle)
4. [Étape 2 : Effectuer des prédictions individuelles](#étape-2--effectuer-des-prédictions-individuelles)
5. [Étape 3 : Terminer le modèle déployé](#étape-3--terminer-le-modèle-déployé)
6. [Étape 4 : Effectuer une transformation par lot](#étape-4--effectuer-une-transformation-par-lot)
7. [Conclusion](#conclusion)

---

## Introduction

### Aperçu

Nous travaillons sur un projet visant à améliorer la détection des anomalies chez les patients orthopédiques en utilisant le machine learning. Nous disposons d'un jeu de données biomécaniques que nous allons utiliser pour entraîner un modèle capable de prédire si un patient présente une anomalie ou non.

### À propos du jeu de données

Le jeu de données provient du **Vertebral Column Dataset** de l'UCI Machine Learning Repository. Il contient des mesures biomécaniques de la colonne vertébrale de patients, avec une étiquette indiquant si le patient est "Normal" ou "Anormal".

---

## Configuration du lab

### Importation des bibliothèques et des données

**Objectif** : Importer les bibliothèques nécessaires et charger les données pour les préparer à l'entraînement du modèle.

```python
# Importation des bibliothèques Python nécessaires
import warnings, requests, zipfile, io
warnings.simplefilter('ignore')  # Ignorer les avertissements pour une sortie plus claire
import pandas as pd  # Pour la manipulation des données
from scipy.io import arff  # Pour lire les fichiers .arff (format de dataset)
import os
import boto3  # AWS SDK pour Python, pour interagir avec S3
import sagemaker  # SDK SageMaker pour Python
from sagemaker.image_uris import retrieve  # Pour récupérer les URI des images Docker de SageMaker
from sklearn.model_selection import train_test_split  # Pour diviser le dataset
```

**Explication** :

- Nous importons plusieurs bibliothèques essentielles pour le traitement des données, l'interaction avec AWS et l'entraînement du modèle.
- Nous configurons les avertissements pour qu'ils soient ignorés afin de rendre les sorties plus lisibles.

### Préparation des données

**Objectif** : Télécharger le jeu de données, le charger dans un DataFrame pandas, et le préparer pour l'entraînement.

```python
# Téléchargement du fichier zip contenant le dataset
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Chargement du fichier .arff dans un DataFrame pandas
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])

# Conversion des étiquettes de classe en valeurs numériques
class_mapper = {b'Abnormal':1, b'Normal':0}
df['class'] = df['class'].replace(class_mapper)

# Réorganisation des colonnes pour placer la classe en premier
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
```

**Explication** :

- Nous téléchargeons le fichier zip du dataset depuis l'UCI Repository.
- Nous extrayons le contenu du zip pour accéder au fichier `.arff`.
- Nous chargeons le fichier `.arff` dans un DataFrame pandas.
- Nous remplaçons les étiquettes de classe (`'Abnormal'` et `'Normal'`) par des valeurs numériques (`1` et `0`).
- Nous réorganisons les colonnes pour que la colonne 'class' soit la première, ce qui est souvent requis pour certains algorithmes d'entraînement.

**Vérification du DataFrame** :

```python
df.head()
```

### Division des données en ensembles d'entraînement, de test et de validation

**Objectif** : Diviser le dataset en trois ensembles pour l'entraînement, le test et la validation du modèle.

```python
# Division du dataset
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])
```

**Explication** :

- Nous utilisons `train_test_split` pour diviser le dataset en 80% pour l'entraînement et 20% pour le test et la validation.
- Nous divisons ensuite les 20% restants en deux parties égales pour le test et la validation.
- La stratification est utilisée pour conserver la même proportion de classes dans chaque ensemble.

### Téléchargement des données sur S3

**Objectif** : Télécharger les ensembles de données préparés sur Amazon S3 pour qu'ils soient accessibles par SageMaker.

```python
# Définition des noms de fichiers et du préfixe S3
prefix = 'lab3'
train_file = 'vertebral_train.csv'
test_file = 'vertebral_test.csv'
validate_file = 'vertebral_validate.csv'

# Fonction pour télécharger un DataFrame en tant que CSV sur S3
s3_resource = boto3.Session().resource('s3')

def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())

# Téléchargement des ensembles sur S3
upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)
```

**Explication** :

- Nous définissons une fonction `upload_s3_csv` qui prend un DataFrame et le télécharge sur S3 en tant que fichier CSV.
- Nous utilisons cette fonction pour télécharger nos ensembles d'entraînement, de test et de validation dans des dossiers séparés sur S3.
- Cela permet à SageMaker d'accéder facilement aux données lors de l'entraînement et du déploiement du modèle.

### Configuration et entraînement du modèle XGBoost

**Objectif** : Configurer les paramètres du modèle XGBoost, créer un Estimator SageMaker, et entraîner le modèle.

```python
# Récupération de l'URI de l'image Docker pour XGBoost
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')

# Définition des hyperparamètres du modèle
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic"
}

# Définition de l'emplacement de sortie sur S3 pour le modèle entraîné
s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)

# Création de l'estimateur XGBoost
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=s3_output_location,
    hyperparameters=hyperparams,
    sagemaker_session=sagemaker.Session()
)

# Configuration des canaux de données pour l'entraînement et la validation
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix, train_file),
    content_type='text/csv'
)

validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix, validate_file),
    content_type='text/csv'
)

data_channels = {'train': train_channel, 'validation': validate_channel}

# Entraînement du modèle
xgb_model.fit(inputs=data_channels, logs=False)
```

**Explication** :

- **Récupération de l'image Docker** : Nous obtenons l'URI de l'image Docker pour XGBoost compatible avec SageMaker.
- **Hyperparamètres** : Nous définissons les hyperparamètres du modèle, notamment :
  - `num_round` : Le nombre d'itérations d'entraînement.
  - `eval_metric` : La métrique d'évaluation utilisée (ici, l'AUC).
  - `objective` : L'objectif d'entraînement (ici, une classification binaire avec sortie probabiliste).
- **Estimator** : Nous créons un objet Estimator de SageMaker en spécifiant l'image Docker, le rôle IAM, le type d'instance, l'emplacement de sortie et les hyperparamètres.
- **Canaux de données** : Nous configurons les canaux pour l'entraînement et la validation en pointant vers les emplacements S3 de nos données.
- **Entraînement** : Nous lançons l'entraînement du modèle avec la méthode `fit()`. Le paramètre `logs=False` désactive l'affichage des logs détaillés.

---

## Étape 1 : Hébergement du modèle

**Objectif** : Déployer le modèle entraîné sur un endpoint SageMaker pour pouvoir effectuer des prédictions en temps réel.

```python
# Déploiement du modèle sur un endpoint
xgb_predictor = xgb_model.deploy(
    initial_instance_count=1,
    serializer=sagemaker.serializers.CSVSerializer(),
    instance_type='ml.m4.xlarge'
)
```

**Explication** :

- Nous utilisons la méthode `deploy()` sur notre objet `xgb_model` pour déployer le modèle sur un endpoint SageMaker.
- **Paramètres** :
  - `initial_instance_count` : Le nombre d'instances ML à utiliser (ici, 1).
  - `serializer` : Spécifie comment les données d'entrée seront sérialisées (converties en format approprié). Nous utilisons un sérialiseur CSV car nos données sont au format CSV.
  - `instance_type` : Le type d'instance ML à utiliser pour le déploiement.

**Ce que fait SageMaker** :

- Crée un modèle hébergé sur une instance ML.
- Configure un endpoint accessible pour envoyer des requêtes de prédiction.
- Gère l'infrastructure sous-jacente pour vous.

---

## Étape 2 : Effectuer des prédictions individuelles

**Objectif** : Envoyer des données au modèle déployé et obtenir des prédictions.

### Préparation des données de test

**Sélection d'une ligne de données pour la prédiction** :

```python
# Affichage de la forme de l'ensemble de test
print(test.shape)
```

**Explication** :

- Nous avons 31 instances dans notre ensemble de test.

```python
# Affichage des premières lignes de l'ensemble de test
test.head(5)
```

**Explication** :

- Nous examinons les premières lignes pour comprendre la structure des données.
- Nous remarquons que la première colonne est 'class', suivie des caractéristiques.

**Préparation de la ligne de données** :

```python
# Sélection de la première ligne sans la colonne 'class'
row = test.iloc[0:1, 1:]
row.head()
```

**Explication** :

- Nous utilisons `iloc` pour sélectionner la première ligne (`0:1`) et toutes les colonnes à partir de la deuxième (`1:`) pour exclure la colonne 'class'.
- Ceci représente les caractéristiques de notre instance sans l'étiquette.

### Conversion de la ligne en format CSV

```python
# Conversion de la ligne en une chaîne CSV
batch_X_csv_buffer = io.StringIO()
row.to_csv(batch_X_csv_buffer, header=False, index=False)
test_row = batch_X_csv_buffer.getvalue()
print(test_row)
```

**Explication** :

- Nous écrivons la ligne dans un buffer CSV en mémoire.
- Nous récupérons la valeur du buffer en tant que chaîne de caractères.
- Ceci est nécessaire car le prédicteur attend une entrée sous forme de chaîne CSV.

### Envoi de la requête de prédiction

```python
# Envoi de la requête au modèle déployé
prediction = xgb_predictor.predict(test_row)
print(prediction)
```

**Explication** :

- Nous utilisons la méthode `predict()` du prédicteur en lui passant notre chaîne CSV.
- Le modèle renvoie une probabilité sous forme de chaîne de bytes (par exemple, `b'0.9966071844100952'`).

### Interprétation du résultat

**Comparaison avec la valeur réelle** :

```python
# Affichage de la valeur réelle de 'class' pour la ligne sélectionnée
print(test.iloc[0]['class'])
```

**Explication** :

- Nous accédons à la valeur de la colonne 'class' pour la première ligne de notre ensemble de test.
- Cela nous permet de comparer la prédiction du modèle avec la valeur réelle.

**Challenge facultatif** :

- Essayez de modifier l'index dans `iloc` pour sélectionner d'autres lignes et voir si le modèle prédit correctement.
- Exemple : `row = test.iloc[1:2, 1:]`

---

## Étape 3 : Terminer le modèle déployé

**Objectif** : Supprimer le endpoint déployé pour éviter des coûts inutiles.

```python
# Suppression du endpoint
xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```

**Explication** :

- Nous appelons `delete_endpoint()` sur notre prédicteur.
- Le paramètre `delete_endpoint_config=True` assure que la configuration du endpoint est également supprimée.
- Cela libère les ressources associées au endpoint.

---

## Étape 4 : Effectuer une transformation par lot

**Objectif** : Utiliser le Batch Transform de SageMaker pour effectuer des prédictions sur l'ensemble complet de test.

### Préparation des données pour le Batch Transform

```python
# Sélection de toutes les lignes et des colonnes de caractéristiques
batch_X = test.iloc[:, 1:]
batch_X.head()
```

**Explication** :

- Nous prenons tout l'ensemble de test et retirons la colonne 'class' pour ne garder que les caractéristiques.

### Téléchargement des données sur S3

```python
# Nom du fichier d'entrée pour le batch
batch_X_file = 'batch-in.csv'

# Téléchargement du fichier sur S3
upload_s3_csv(batch_X_file, 'batch-in', batch_X)
```

**Explication** :

- Nous enregistrons les données de test dans un fichier CSV.
- Nous utilisons la fonction `upload_s3_csv` précédemment définie pour le télécharger sur S3 dans le dossier 'batch-in'.

### Configuration du Batch Transform

```python
# Emplacements S3 pour l'entrée et la sortie du Batch Transform
batch_output = "s3://{}/{}/batch-out/".format(bucket, prefix)
batch_input = "s3://{}/{}/batch-in/{}".format(bucket, prefix, batch_X_file)

# Création du transformateur à partir du modèle entraîné
xgb_transformer = xgb_model.transformer(
    instance_count=1,
    instance_type='ml.m4.xlarge',
    strategy='MultiRecord',
    assemble_with='Line',
    output_path=batch_output
)
```

**Explication** :

- Nous spécifions où se trouvent les données d'entrée et où seront stockés les résultats sur S3.
- Nous créons un objet transformeur (`xgb_transformer`) à partir de notre modèle entraîné.
- **Paramètres importants** :
  - `strategy='MultiRecord'` : Permet de traiter plusieurs enregistrements à la fois.
  - `assemble_with='Line'` : Spécifie comment les résultats seront assemblés (ici, ligne par ligne).
  - `output_path` : Emplacement S3 où les résultats seront stockés.

### Exécution du Batch Transform

```python
# Lancement du Batch Transform
xgb_transformer.transform(
    data=batch_input,
    data_type='S3Prefix',
    content_type='text/csv',
    split_type='Line'
)

# Attente de la fin du travail
xgb_transformer.wait()
```

**Explication** :

- Nous appelons la méthode `transform()` en passant les paramètres nécessaires.
- **Paramètres** :
  - `data` : Emplacement des données d'entrée.
  - `data_type='S3Prefix'` : Indique que les données sont sur S3.
  - `content_type='text/csv'` : Type de contenu des données.
  - `split_type='Line'` : Les données sont séparées par des lignes.
- `xgb_transformer.wait()` : Attend la fin du travail de transformation.

**Ce que fait SageMaker** :

- Lance une instance ML avec le modèle déployé.
- Applique le modèle sur chaque enregistrement de l'entrée.
- Stocke les prédictions dans l'emplacement spécifié sur S3.
- Termine l'instance une fois le travail terminé.

### Téléchargement et analyse des résultats

**Téléchargement des prédictions depuis S3** :

```python
# Récupération des prédictions depuis S3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket=bucket, Key="{}/batch-out/{}".format(prefix, 'batch-in.csv.out'))
target_predicted = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=',', names=['class'])
target_predicted.head()
```

**Explication** :

- Nous utilisons le client S3 pour accéder au fichier de sortie généré par le Batch Transform.
- Nous lisons le fichier CSV des prédictions dans un DataFrame pandas.

### Conversion des probabilités en classes binaires

**Définition de la fonction de conversion** :

```python
# Fonction pour convertir les probabilités en classes binaires
def binary_convert(x):
    threshold = 0.65  # Seuil de décision
    if x > threshold:
        return 1
    else:
        return 0
```

**Explication** :

- Cette fonction prend une probabilité et la convertit en `1` si elle est supérieure au seuil, sinon `0`.
- Le seuil de 0.65 peut être ajusté en fonction de la performance souhaitée.

**Application de la fonction** :

```python
# Application de la fonction aux prédictions
target_predicted['binary'] = target_predicted['class'].apply(binary_convert)
```

**Comparaison avec les valeurs réelles** :

```python
# Affichage des prédictions binaires
print(target_predicted.head())

# Affichage des valeurs réelles
print(test['class'].reset_index(drop=True).head())
```

**Explication** :

- Nous comparons les classes prédites avec les classes réelles pour évaluer la performance du modèle.

**Challenge facultatif** :

- Expérimentez en modifiant le seuil dans la fonction `binary_convert` pour voir comment cela affecte les résultats.
- Essayez d'autres métriques ou méthodes pour évaluer la performance du modèle.

---

## Conclusion

Dans ce guide, nous avons :

- Importé et préparé un jeu de données biomécaniques pour la classification binaire.
- Entraîné un modèle XGBoost en utilisant Amazon SageMaker.
- Déployé le modèle sur un endpoint pour des prédictions en temps réel.
- Effectué des prédictions individuelles et interprété les résultats.
- Supprimé le endpoint pour gérer les ressources efficacement.
- Utilisé le Batch Transform de SageMaker pour effectuer des prédictions sur un ensemble complet de données.
- Converti les probabilités de sortie en classes binaires pour l'évaluation.

**Points clés à retenir** :

- **Préparation des données** : Une étape cruciale qui influence la qualité du modèle.
- **Gestion des ressources** : Toujours supprimer les ressources inutilisées pour éviter des coûts supplémentaires.
- **Évaluation du modèle** : Utiliser des métriques appropriées pour évaluer la performance et ajuster les hyperparamètres en conséquence.
- **SageMaker** : Offre des outils puissants pour entraîner, déployer et évaluer des modèles de machine learning à grande échelle.

---

**Prochaines étapes** :

- Calculer des métriques d'évaluation pour quantifier la performance du modèle.
- Expérimenter avec l'ajustement des hyperparamètres pour améliorer les prédictions.
- Explorer d'autres algorithmes ou techniques de feature engineering pour potentiellement améliorer les résultats.

---

**Félicitations !** Vous avez terminé ce guide détaillé. Vous êtes maintenant mieux équipé pour travailler avec Amazon SageMaker et le machine learning en général.
