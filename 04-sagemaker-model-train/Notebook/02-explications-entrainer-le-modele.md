-------------------
# Explications entraîner le modèle
-------------------

## Étapes:



1. **Importation et chargement des données** : Téléchargement et extraction des données, conversion en DataFrame et mappage des valeurs de classe.
2. **Exploration des données** : Exploration des dimensions et des colonnes.
3. **Préparation des données** : Déplacement de la colonne cible, division des données en ensembles d'entraînement, de validation et de test.
4. **Chargement dans Amazon S3** : Configuration de l'emplacement S3 et chargement des ensembles de données sous forme de fichiers CSV.
5. **Configuration et entraînement du modèle XGBoost avec Amazon SageMaker** : Configuration des hyperparamètres et lancement de l’entraînement.



## Code

```python
# Importer les bibliothèques nécessaires
import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff
import boto3

warnings.simplefilter('ignore')

# Télécharger et extraire le fichier de données
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Charger le fichier .arff et le convertir en DataFrame Pandas
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])

# Mapper les valeurs de la classe en valeurs numériques
class_mapper = {b'Abnormal': 1, b'Normal': 0}
df['class'] = df['class'].replace(class_mapper)

# Dimensions du jeu de données
df.shape

# Afficher les noms des colonnes
df.columns

# Déplacer la colonne cible en première position
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# Vérifier le nouvel ordre des colonnes
df.columns

# Diviser les données en ensembles d'entraînement et de validation/test
from sklearn.model_selection import train_test_split
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

# Diviser l'ensemble de validation/test en deux
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])

# Vérifier les dimensions de chaque ensemble
print(train.shape)
print(test.shape)
print(validate.shape)

# Compter les occurrences de chaque classe dans chaque ensemble
print(train['class'].value_counts())
print(test['class'].value_counts())
print(validate['class'].value_counts())

# Définir les paramètres S3
bucket = 'c124417a3052642l8225608t1w975050101910-labbucket-qgynmzp5xcwe'
prefix = 'lab3'
train_file = 'vertebral_train.csv'
test_file = 'vertebral_test.csv'
validate_file = 'vertebral_validate.csv'

# Initialiser la ressource S3
s3_resource = boto3.Session().resource('s3')

# Fonction pour télécharger les fichiers CSV vers S3
def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())

# Télécharger les ensembles de données vers S3
upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)

# Obtenir l'URI du conteneur XGBoost
from sagemaker.image_uris import retrieve
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')

# Définir les hyperparamètres pour l'entraînement
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic"
}

# Définir l'emplacement de sortie des modèles dans S3
import sagemaker
s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)

# Configurer l'estimateur XGBoost
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=s3_output_location,
    hyperparameters=hyperparams,
    sagemaker_session=sagemaker.Session()
)

# Définir les canaux d'entraînement et de validation
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix, train_file),
    content_type='text/csv'
)

validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix, validate_file),
    content_type='text/csv'
)

data_channels = {'train': train_channel, 'validation': validate_channel}

# Lancer l'entraînement du modèle
xgb_model.fit(inputs=data_channels, logs=False)
```

------------------
## Explications
------------------

#### Importer les bibliothèques nécessaires

```python
import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff
import boto3
```

1. **warnings** : Cette bibliothèque gère les messages d’avertissement. Nous allons les désactiver pour rendre la sortie du programme plus propre.
2. **requests** : Utilisée pour faire des requêtes HTTP, ce qui nous permet de télécharger le fichier de données en ligne.
3. **zipfile** : Permet de manipuler les fichiers compressés `.zip`.
4. **io** : Fournit des outils pour gérer les opérations d'entrée et de sortie (comme les fichiers en mémoire).
5. **pandas** : Utilisée pour la manipulation et l'analyse des données, particulièrement efficace pour travailler avec des tableaux de données (DataFrames).
6. **scipy.io.arff** : Un module de SciPy pour lire les fichiers `.arff`, un format de données utilisé en machine learning.
7. **boto3** : Bibliothèque de Python pour interagir avec Amazon Web Services (AWS), ici utilisée pour Amazon S3.

---

```python
warnings.simplefilter('ignore')
```

Nous désactivons les messages d’avertissement pour que l’affichage reste simple.

---

#### Télécharger et extraire le fichier de données

```python
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()
```

1. **f_zip** : Lien direct vers le fichier `.zip` contenant notre jeu de données.
2. **requests.get(f_zip, stream=True)** : Envoie une requête HTTP pour télécharger le fichier.
3. **io.BytesIO(r.content)** : Stocke le fichier téléchargé en mémoire au lieu de l’enregistrer sur le disque dur.
4. **zipfile.ZipFile(io.BytesIO(r.content))** : Ouvre le fichier `.zip` en mémoire.
5. **Vertebral_zip.extractall()** : Décompresse le contenu du fichier `.zip` dans le répertoire de travail actuel.

---

#### Charger le fichier `.arff` et le convertir en DataFrame Pandas

```python
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])
```

1. **arff.loadarff('column_2C_weka.arff')** : Charge le fichier `.arff` en un format compatible pour être manipulé en Python.
2. **pd.DataFrame(data[0])** : Crée un DataFrame Pandas à partir des données chargées pour faciliter leur manipulation.

---

#### Mapper les valeurs de la classe en valeurs numériques

```python
class_mapper = {b'Abnormal': 1, b'Normal': 0}
df['class'] = df['class'].replace(class_mapper)
```

1. **class_mapper** : Un dictionnaire qui associe la classe `Abnormal` à `1` et `Normal` à `0`. Les valeurs sont en `bytes` (ex. `b'Abnormal'`) car le fichier `.arff` les charge ainsi.
2. **df['class'].replace(class_mapper)** : Remplace les valeurs textuelles dans la colonne `class` par leurs équivalents numériques définis dans `class_mapper`.

---

#### Dimensions du jeu de données

```python
df.shape
```

Renvoie le nombre de lignes et de colonnes du DataFrame, par exemple `(310, 7)`.

---

#### Afficher les noms des colonnes

```python
df.columns
```

Renvoie les noms de toutes les colonnes dans le DataFrame pour nous assurer qu’elles sont bien chargées.

---

#### Déplacer la colonne cible en première position

```python
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
```

1. **df.columns.tolist()** : Crée une liste de noms de colonnes.
2. **cols[-1:] + cols[:-1]** : Change l'ordre de la liste en mettant la dernière colonne en premier.
3. **df[cols]** : Réorganise le DataFrame selon cet ordre de colonnes.

---

#### Vérifier le nouvel ordre des colonnes

```python
df.columns
```

Vérifie que la colonne cible est maintenant en première position dans le DataFrame.

---

#### Diviser les données en ensembles d'entraînement et de validation/test

```python
from sklearn.model_selection import train_test_split
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
```

1. **train_test_split** : Fonction de scikit-learn qui divise les données en sous-ensembles.
   - **test_size=0.2** : 20 % des données seront affectées à `test_and_validate`.
   - **random_state=42** : Assure que la division sera la même à chaque exécution.
   - **stratify=df['class']** : Maintient la même proportion de classes dans chaque sous-ensemble.

---

#### Diviser l'ensemble de validation/test en deux

```python
test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])
```

Nous séparons `test_and_validate` en deux ensembles, `test` et `validate`, chacun avec 50 % des données de `test_and_validate`.

---

#### Vérifier les dimensions de chaque ensemble

```python
print(train.shape)
print(test.shape)
print(validate.shape)
```

Affiche les dimensions de `train`, `test`, et `validate` pour vérifier que la division a été effectuée correctement.

---

#### Compter les occurrences de chaque classe dans chaque ensemble

```python
print(train['class'].value_counts())
print(test['class'].value_counts())
print(validate['class'].value_counts())
```

Affiche le nombre de valeurs `0` et `1` dans chaque ensemble pour vérifier l’équilibre des classes.

---

#### Définir les paramètres S3

```python
bucket = 'c124417a3052642l8225608t1w975050101910-labbucket-qgynmzp5xcwe'
prefix = 'lab3'
train_file = 'vertebral_train.csv'
test_file = 'vertebral_test.csv'
validate_file = 'vertebral_validate.csv'
```

1. **bucket** : Nom du seau S3 dans lequel les fichiers seront stockés.
2. **prefix** : Préfixe pour organiser les fichiers dans le seau.
3. **train_file, test_file, validate_file** : Noms des fichiers CSV qui seront créés pour chaque ensemble.

---

#### Initialiser la ressource S3

```python
s3_resource = boto3.Session().resource('s3')
```

Crée une connexion à S3 pour pouvoir y envoyer les fichiers.

---

#### Fonction pour télécharger les fichiers CSV vers S3

```python
def upload_s3_csv(filename, folder, dataframe):
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(os.path.join(prefix, folder, filename)).put(Body=csv_buffer.getvalue())
```

1. **csv_buffer = io.StringIO()** : Crée un tampon (buffer) pour enregistrer le fichier en mémoire.
2. **dataframe.to_csv(...)** : Convertit le DataFrame en format CSV.
3. **s3_resource.Bucket(bucket).Object(...).put(...)** : Envoie le CSV vers S3.

---

#### Télécharger les ensembles de données vers S3

```python
upload_s3_csv(train_file, 'train', train)
upload_s3_csv(test_file, 'test', test)
upload_s3_csv(validate_file, 'validate', validate)
```

Appelle la fonction `upload_s3_csv` pour chaque fichier (train, test, validate).

---

#### Obtenir l'URI du conteneur XGBoost

```python
from sagemaker.image_uris import retrieve
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')
```

Récupère l'URI de l’image Docker pour XGBoost dans la région AWS en cours.

---

#### Définir les hyperparamètres pour l'entraînement

```python
hyperparams = {
    "num_round": "42",
    "eval_metric": "auc",
    "objective": "binary:logistic"
}
```

1. **num_round** : Nombre de cycles d'entraînement.
2. **eval_metric** : Métrique d'évaluation, ici `auc` (aire sous la courbe ROC).
3. **objective** : Type de tâche, ici `binary:logistic` pour la classification binaire.

---

#### Définir l'emplacement de sortie des modèles dans S3

```python
import sagemaker
s3_output_location = "s3://{}/{}/output/".format(bucket, prefix)
```

Définit où le modèle sera stocké après l’entraînement.

---

#### Configurer l'estimateur XGBoost

```python
xgb_model = sagemaker.estimator.Estimator(
    container,
    sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=s3_output_location,
    hyperparameters=hyperparams

,
    sagemaker_session=sagemaker.Session()
)
```

1. **container** : URI du conteneur XGBoost.
2. **instance_count** et **instance_type** : Paramètres de configuration de l’instance.
3. **output_path** : Emplacement pour enregistrer le modèle entraîné.

---

#### Définir les canaux d'entraînement et de validation

```python
train_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/train/".format(bucket, prefix, train_file),
    content_type='text/csv'
)

validate_channel = sagemaker.inputs.TrainingInput(
    "s3://{}/{}/validate/".format(bucket, prefix, validate_file),
    content_type='text/csv'
)

data_channels = {'train': train_channel, 'validation': validate_channel}
```

Spécifie les emplacements S3 des fichiers d'entraînement et de validation, qui seront utilisés pour nourrir le modèle pendant l'entraînement.

---

#### Lancer l'entraînement du modèle

```python
xgb_model.fit(inputs=data_channels, logs=False)
```

Démarre l'entraînement du modèle XGBoost avec les ensembles de données spécifiés.


