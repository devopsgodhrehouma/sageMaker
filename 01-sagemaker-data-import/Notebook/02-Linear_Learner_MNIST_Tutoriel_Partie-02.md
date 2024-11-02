-------------------
# Introduction à Amazon SageMaker Linear Learner avec MNIST
-------------------

*Ce tutoriel vous guide dans la configuration et l'entraînement d'un modèle de classification binaire avec Amazon SageMaker pour prédire si une image représente le chiffre 0 ou non.*


### Prédiction binaire pour déterminer si un chiffre manuscrit est un 0

Ce tutoriel présente l'algorithme **Linear Learner** d'Amazon SageMaker pour effectuer une classification binaire à partir du jeu de données MNIST. Nous allons prédire si une image de chiffre manuscrit représente un 0 ou un autre chiffre (1, 2, 3, ... 9) en utilisant un modèle linéaire binaire.

### Introduction

Nous allons travailler avec le jeu de données MNIST, constitué d'images 28 x 28 en niveaux de gris. Le modèle Linear Learner est un algorithme de **classification supervisée** qui apprend une fonction linéaire pour associer chaque image à un label 0 (pour les 0) ou 1 (pour tous les autres chiffres). Cet algorithme entraîne plusieurs modèles en parallèle avec différents hyperparamètres pour optimiser la précision tout en étant efficace.

### Prérequis et Prétraitement

1. **Permissions et Variables d'Environnement**

   Assurez-vous d'avoir installé le SDK SageMaker :

   ```python
   !pip install --upgrade sagemaker
   ```

   Importez les bibliothèques nécessaires et configurez le rôle IAM et les buckets S3 :

   ```python
   import boto3
   import sagemaker
   from sagemaker import get_execution_role

   sess = sagemaker.Session()
   region = boto3.Session().region_name
   downloaded_data_bucket = f"sagemaker-example-files-prod-{region}"
   downloaded_data_prefix = "datasets/image/MNIST"
   bucket = sess.default_bucket()
   prefix = "sagemaker/DEMO-linear-mnist"
   role = get_execution_role()
   ```

-----------------------------
# Explication 1
-----------------------------

```python
import boto3
import sagemaker
from sagemaker import get_execution_role
```

1. **`import boto3`** : Cette ligne importe le module `boto3`, qui est le SDK (Software Development Kit) pour interagir avec les services AWS en Python. `boto3` permet de créer, configurer et gérer les ressources et services AWS, comme S3, EC2, et SageMaker.

2. **`import sagemaker`** : Cette ligne importe la bibliothèque `sagemaker`, qui est un SDK conçu pour faciliter l’utilisation d’Amazon SageMaker. `sagemaker` offre des outils pour configurer et déployer des modèles de machine learning facilement sur AWS SageMaker.

3. **`from sagemaker import get_execution_role`** : Cette instruction importe la fonction `get_execution_role` depuis la bibliothèque `sagemaker`. Cette fonction est utilisée pour récupérer le rôle IAM (Identity and Access Management) qui a les permissions nécessaires pour exécuter les tâches dans SageMaker. Ce rôle est essentiel pour que SageMaker puisse accéder aux ressources AWS comme les buckets S3 et les instances de calcul.


```python
sess = sagemaker.Session()
```

4. **`sess = sagemaker.Session()`** : Cette ligne crée une session SageMaker en utilisant `sagemaker.Session()`. Une session est une instance de la classe `Session` de la bibliothèque `sagemaker`, qui représente la connexion à SageMaker dans votre région AWS actuelle. Elle gère également des informations contextuelles sur les configurations par défaut, comme le bucket S3 à utiliser pour stocker les données.

```python
region = boto3.Session().region_name
```

5. **`region = boto3.Session().region_name`** : Cette ligne crée une session avec `boto3` pour obtenir la région AWS actuelle (`region_name`). La région est le centre de données AWS où les ressources seront créées ou gérées (par exemple, "us-west-2" pour l'ouest des États-Unis). Ce code permet de récupérer la région sans la spécifier manuellement, ce qui peut être utile pour des configurations dynamiques.

```python
downloaded_data_bucket = f"sagemaker-example-files-prod-{region}"
```

6. **`downloaded_data_bucket = f"sagemaker-example-files-prod-{region}"`** : Cette ligne définit le nom du bucket S3 source dans lequel les fichiers d’exemple sont stockés. Le nom du bucket est construit dynamiquement en ajoutant le nom de la région à un préfixe (`sagemaker-example-files-prod-`). Cela signifie que pour chaque région AWS, il existe un bucket avec les fichiers d'exemples nécessaires à SageMaker, permettant de télécharger les données indépendamment de la région.

```python
downloaded_data_prefix = "datasets/image/MNIST"
```

7. **`downloaded_data_prefix = "datasets/image/MNIST"`** : Cette ligne définit le chemin (ou préfixe) de l’emplacement des données d'exemple dans le bucket. Ici, le chemin mène au dossier `datasets/image/MNIST` dans le bucket. Ce chemin contient les données du jeu de données MNIST (images de chiffres manuscrits), couramment utilisées dans les exemples de machine learning.

```python
bucket = sess.default_bucket()
```

8. **`bucket = sess.default_bucket()`** : Cette ligne crée un bucket S3 par défaut dans la région actuelle de SageMaker (si ce bucket n'existe pas déjà). SageMaker utilise ce bucket pour stocker les données et les résultats d'entraînement par défaut. Cette instruction facilite la gestion du stockage S3 en évitant de définir manuellement un bucket spécifique.

```python
prefix = "sagemaker/DEMO-linear-mnist"
```

9. **`prefix = "sagemaker/DEMO-linear-mnist"`** : Cette ligne définit un préfixe pour l'emplacement dans le bucket S3 où les données et les résultats de l’entraînement seront stockés. `sagemaker/DEMO-linear-mnist` est un dossier dans le bucket par défaut de SageMaker où les fichiers de cet exemple spécifique (en l’occurrence, le modèle MNIST pour une régression linéaire) seront sauvegardés.

```python
role = get_execution_role()
```

10. **`role = get_execution_role()`** : Cette ligne appelle la fonction `get_execution_role()` pour obtenir le rôle IAM (Identity and Access Management) qui est attaché au notebook ou à l’instance SageMaker actuelle. Ce rôle détermine les permissions que SageMaker peut utiliser pour interagir avec d’autres services AWS (comme S3) pendant l’entraînement ou le déploiement du modèle.

En somme, ces lignes de code configurent et préparent l’environnement SageMaker pour gérer les données et les ressources S3, ainsi que le rôle d'exécution qui permettra à SageMaker d'interagir en toute sécurité avec les services AWS pendant l'entraînement du modèle.


2. **Chargement des données**

   Téléchargez les données MNIST et chargez-les en mémoire :

   ```python
   import pickle, gzip, numpy

   s3 = boto3.client("s3")
   s3.download_file(downloaded_data_bucket, f"{downloaded_data_prefix}/mnist.pkl.gz", "mnist.pkl.gz")

   with gzip.open("mnist.pkl.gz", "rb") as f:
       train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
   ```

3. **Inspection des données**

   Visualisez un échantillon d’image pour vérifier le contenu :

   ```python
   import matplotlib.pyplot as plt
   %matplotlib inline

   def show_digit(img, caption="", subplot=None):
       if subplot is None:
           _, (subplot) = plt.subplots(1, 1)
       imgr = img.reshape((28, 28))
       subplot.axis("off")
       subplot.imshow(imgr, cmap="gray")
       plt.title(caption)

   show_digit(train_set[0][30], f"C'est un {train_set[1][30]}")
   ```

4. **Conversion des données**

   Convertissez les données en format `recordIO` compatible avec SageMaker :

   ```python
   import io
   import numpy as np
   import sagemaker.amazon.common as smac

   train_set_vectors = np.array([t.tolist() for t in train_set[0]]).astype("float32")
   train_set_labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype("float32")

   validation_set_vectors = np.array([t.tolist() for t in valid_set[0]]).astype("float32")
   validation_set_labels = np.where(np.array([t.tolist() for t in valid_set[1]]) == 0, 1, 0).astype("float32")

   train_set_buf = io.BytesIO()
   validation_set_buf = io.BytesIO()
   smac.write_numpy_to_dense_tensor(train_set_buf, train_set_vectors, train_set_labels)
   smac.write_numpy_to_dense_tensor(validation_set_buf, validation_set_vectors, validation_set_labels)

   train_set_buf.seek(0)
   validation_set_buf.seek(0)
   ```

5. **Téléchargement vers S3**

   Chargez les données préparées dans S3 pour l'entraînement :

   ```python
   import os

   key = "recordio-pb-data"
   boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "train", key)).upload_fileobj(train_set_buf)
   boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "validation", key)).upload_fileobj(validation_set_buf)

   s3_train_data = f"s3://{bucket}/{prefix}/train/{key}"
   s3_validation_data = f"s3://{bucket}/{prefix}/validation/{key}"

   output_location = f"s3://{bucket}/{prefix}/output"
   ```

### Entraînement du Modèle Linéaire

1. **Configuration de l'entraînement**

   Initialisez le conteneur et l'estimateur SageMaker pour Linear Learner :

   ```python
   from sagemaker import image_uris

   container = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")
   linear = sagemaker.estimator.Estimator(
       container,
       role,
       instance_count=1,
       instance_type="ml.c4.xlarge",
       output_path=output_location,
       sagemaker_session=sess,
   )
   linear.set_hyperparameters(feature_dim=784, predictor_type="binary_classifier", mini_batch_size=200)
   linear.fit({"train": s3_train_data})
   ```

2. **Optimisation Automatique des Hyperparamètres**

   Pour améliorer la précision, utilisez la **tuning automatique des hyperparamètres** :

   ```python
   from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

   hyperparameter_ranges = {
       "wd": ContinuousParameter(1e-7, 1),
       "learning_rate": ContinuousParameter(1e-5, 1),
       "mini_batch_size": IntegerParameter(100, 2000),
   }

   hp_tuner = HyperparameterTuner(
       linear,
       "validation:binary_f_beta",
       hyperparameter_ranges,
       max_jobs=6,
       max_parallel_jobs=2,
       objective_type="Maximize",
   )
   hp_tuner.fit(inputs={"train": s3_train_data, "validation": s3_validation_data})
   ```

### Déploiement et Validation du Modèle

1. **Déploiement de l’Endpoint**

   Créez un endpoint pour héberger le modèle :

   ```python
   linear_predictor = hp_tuner.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
   ```

2. **Validation des Prédictions**

   Validez le modèle avec une prédiction unique et un ensemble de test complet :

   ```python
   from sagemaker.serializers import CSVSerializer
   from sagemaker.deserializers import JSONDeserializer

   linear_predictor.serializer = CSVSerializer()
   linear_predictor.deserializer = JSONDeserializer()

   result = linear_predictor.predict(train_set[0][30:31], initial_args={"ContentType": "text/csv"})
   print(result)

   # Prédictions en lot
   predictions = []
   for array in np.array_split(test_set[0], 100):
       result = linear_predictor.predict(array)
       predictions += [r["predicted_label"] for r in result["predictions"]]
   ```

3. **Évaluation du Modèle**

   Calculez la **matrice de confusion** pour évaluer la précision :

   ```python
   import pandas as pd

   pd.crosstab(
       np.where(test_set[1] == 0, 1, 0),
       predictions,
       rownames=["réels"],
       colnames=["prédictions"]
   )
   ```

### (Optionnel) Suppression de l'Endpoint

Pour économiser les ressources :

```python
linear_predictor.delete_model()
linear_predictor.delete_endpoint()
```
