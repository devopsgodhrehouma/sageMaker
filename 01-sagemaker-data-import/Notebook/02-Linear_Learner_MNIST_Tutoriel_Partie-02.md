# Notebook2

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













-----------------------------
-----------------------------
-----------------------------
-----------------------------
-----------------------------


# 2. **Chargement des données**

   Téléchargez les données MNIST et chargez-les en mémoire :

   ```python
   import pickle, gzip, numpy

   s3 = boto3.client("s3")
   s3.download_file(downloaded_data_bucket, f"{downloaded_data_prefix}/mnist.pkl.gz", "mnist.pkl.gz")

   with gzip.open("mnist.pkl.gz", "rb") as f:
       train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
   ```

-----------
# Explication
----------

```python
import pickle, gzip, numpy
```

1. **`import pickle, gzip, numpy`** :
   - **`pickle`** : Le module `pickle` est une bibliothèque Python utilisée pour sérialiser et désérialiser des objets Python. La sérialisation (ou "pickling") permet de sauvegarder un objet dans un fichier pour le réutiliser ultérieurement, tandis que la désérialisation (ou "unpickling") permet de le charger.
   - **`gzip`** : Ce module permet de manipuler des fichiers compressés au format GZIP, qui est couramment utilisé pour réduire la taille des fichiers.
   - **`numpy`** : NumPy est une bibliothèque Python essentielle pour les calculs scientifiques et le traitement de données. Elle est notamment utilisée pour travailler avec des tableaux (arrays) de grande taille, ce qui est commun en machine learning et en data science.

```python
s3 = boto3.client("s3")
```

2. **`s3 = boto3.client("s3")`** :
   - Cette ligne crée un client S3 en utilisant `boto3.client("s3")`. Le client S3 permet d'interagir avec le service S3 d’AWS, en téléchargeant et en téléchargeant des fichiers, en listant les objets, en gérant les buckets, etc.
   - Ici, on initialise ce client pour exécuter des actions spécifiques sur le service S3.

```python
s3.download_file(downloaded_data_bucket, f"{downloaded_data_prefix}/mnist.pkl.gz", "mnist.pkl.gz")
```

3. **`s3.download_file(downloaded_data_bucket, f"{downloaded_data_prefix}/mnist.pkl.gz", "mnist.pkl.gz")`** :
   - Cette ligne utilise le client S3 pour télécharger un fichier spécifique depuis un bucket S3.
   - **`downloaded_data_bucket`** : Le nom du bucket S3 où le fichier est stocké (défini précédemment).
   - **`f"{downloaded_data_prefix}/mnist.pkl.gz"`** : Le chemin complet du fichier dans le bucket S3. `downloaded_data_prefix` est le préfixe du chemin, auquel on ajoute le nom du fichier, ici `mnist.pkl.gz`.
   - **`"mnist.pkl.gz"`** : Le nom sous lequel le fichier sera sauvegardé localement après le téléchargement.
   - Au final, cette commande télécharge le fichier `mnist.pkl.gz` dans le répertoire local pour le traitement.

```python
with gzip.open("mnist.pkl.gz", "rb") as f:
```

4. **`with gzip.open("mnist.pkl.gz", "rb") as f`** :
   - Cette ligne ouvre le fichier `mnist.pkl.gz` (compressé au format GZIP) en mode lecture binaire (`"rb"`).
   - **`with`** : Le mot-clé `with` est utilisé ici pour ouvrir le fichier et garantir qu'il sera automatiquement fermé après avoir été lu, même si une erreur survient.
   - **`gzip.open("mnist.pkl.gz", "rb")`** : Décompresse le fichier `mnist.pkl.gz` pour permettre la lecture du contenu.

```python
train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
```

5. **`train_set, valid_set, test_set = pickle.load(f, encoding="latin1")`** :
   - Cette ligne charge les données à partir du fichier décompressé en utilisant `pickle.load`.
   - **`pickle.load(f, encoding="latin1")`** : Cette fonction désérialise le contenu du fichier décompressé et retourne les données dans leur format d'origine. L'argument `encoding="latin1"` est utilisé pour assurer la compatibilité lors du chargement de fichiers créés dans des versions plus anciennes de Python.
   - Les données chargées sont stockées dans trois ensembles : `train_set`, `valid_set`, et `test_set`. Ces variables contiennent respectivement les données d’entraînement, de validation et de test du jeu de données MNIST.

En résumé, ce code télécharge un fichier compressé contenant le jeu de données MNIST depuis un bucket S3, décompresse le fichier, et charge les données pour les utiliser ensuite dans le cadre d’un modèle de machine learning.




-----------------------------
-----------------------------
-----------------------------
-----------------------------
-----------------------------


# 3. **Inspection des données**

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

-------------
# Explication
--------

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

1. **`import matplotlib.pyplot as plt`** :
   - Cette ligne importe `matplotlib.pyplot`, une bibliothèque de Python utilisée pour créer des graphiques et visualiser des données. `pyplot` fournit des fonctions permettant de créer facilement des graphiques et des images, et ici, on l'importe en utilisant l'alias `plt` pour simplifier les appels de fonction.

2. **`%matplotlib inline`** :
   - Ce code est une commande magique utilisée dans les notebooks Jupyter. Elle permet d'afficher directement les graphiques générés dans les cellules du notebook, sans avoir besoin de les afficher dans une nouvelle fenêtre. 

```python
def show_digit(img, caption="", subplot=None):
```

3. **`def show_digit(img, caption="", subplot=None):`** :
   - Cette ligne définit une fonction nommée `show_digit` qui sert à afficher une image d’un chiffre, en ajoutant un titre facultatif (`caption`). Cette fonction prend trois paramètres :
      - **`img`** : Le tableau (array) représentant l'image à afficher.
      - **`caption`** : Un texte optionnel qui apparaîtra comme titre de l'image. Par défaut, il est vide.
      - **`subplot`** : Un objet `subplot` optionnel. S'il n'est pas fourni (c'est-à-dire s'il est `None`), la fonction crée un subplot unique pour afficher l'image.

```python
if subplot is None:
    _, (subplot) = plt.subplots(1, 1)
```

4. **`if subplot is None:`** :
   - Vérifie si un `subplot` n’a pas été fourni lors de l'appel de la fonction. Si aucun `subplot` n'est spécifié (c'est-à-dire si `subplot` est `None`), alors la fonction en crée un.

5. **`_, (subplot) = plt.subplots(1, 1)`** :
   - **`plt.subplots(1, 1)`** crée une figure avec une grille de sous-graphes de 1 ligne et 1 colonne (une seule image). Cela permet d'obtenir un objet `subplot` unique.
   - Le `_` est une variable temporaire pour ignorer la sortie de la figure créée (car seule la variable `subplot` est utile ici).
   - Cette ligne garantit qu'il y a un subplot disponible pour afficher l'image, même si aucun n'est fourni en paramètre.

```python
imgr = img.reshape((28, 28))
```

6. **`imgr = img.reshape((28, 28))`** :
   - Cette ligne redimensionne le tableau `img` en une matrice de 28x28 pixels, correspondant à la taille d'une image dans le jeu de données MNIST. Cette transformation est nécessaire car les images MNIST sont stockées sous forme de vecteurs 1D, et pour afficher l'image, elle doit être restructurée en une forme 2D.

```python
subplot.axis("off")
```

7. **`subplot.axis("off")`** :
   - Désactive les axes autour de l'image pour qu'ils n'apparaissent pas. Cela permet de montrer uniquement l'image sans lignes de grille ou valeurs d'axe, rendant l'affichage plus propre.

```python
subplot.imshow(imgr, cmap="gray")
```

8. **`subplot.imshow(imgr, cmap="gray")`** :
   - Cette ligne affiche l’image dans le subplot créé. `imshow` est une fonction de `matplotlib` pour afficher une matrice 2D sous forme d’image.
   - **`cmap="gray"`** : Spécifie que l'image doit être affichée en niveaux de gris (ce qui est approprié pour MNIST, car les chiffres sont en noir et blanc).

```python
plt.title(caption)
```

9. **`plt.title(caption)`** :
   - Définit le titre de l'image en utilisant le texte fourni dans le paramètre `caption`. Ce texte apparaîtra au-dessus de l’image.

```python
show_digit(train_set[0][30], f"C'est un {train_set[1][30]}")
```

10. **`show_digit(train_set[0][30], f"C'est un {train_set[1][30]}")`** :
    - Cette ligne appelle la fonction `show_digit` pour afficher une image spécifique et lui attribuer un titre.
    - **`train_set[0][30]`** : Cette expression accède à la 30e image dans l’ensemble d’entraînement (`train_set[0]` contient les images).
    - **`f"C'est un {train_set[1][30]}"`** : Utilise une f-string pour créer un titre indiquant la classe ou le chiffre représenté par l’image (par exemple, "C'est un 5"). `train_set[1][30]` contient l’étiquette (ou le chiffre) associé à cette image.

En résumé, ce code permet de visualiser une image de chiffre MNIST avec un titre indiquant sa classe, tout en rendant l'affichage propre (sans axes visibles) et en convertissant l’image en niveaux de gris pour mieux représenter les chiffres.




-----------------------------
-----------------------------
-----------------------------
-----------------------------
-----------------------------


# 4. **Conversion des données**

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

-----------
# Explication
----------


```python
import io
import numpy as np
import sagemaker.amazon.common as smac
```

1. **`import io`** :
   - `io` est une bibliothèque Python permettant de gérer les flux de données (streams) en mémoire ou en fichiers. Ici, `io.BytesIO` sera utilisé pour créer des flux de données en mémoire (sous forme binaire), permettant d'écrire et de lire des données dans un format compatible avec les modèles Amazon SageMaker.

2. **`import numpy as np`** :
   - Cette ligne importe `numpy`, une bibliothèque pour le calcul scientifique en Python, en l'abrégeant en `np`. `numpy` est utilisée ici pour créer et manipuler des tableaux (arrays) de données, notamment pour préparer les données d'entraînement et de validation en tant que matrices numériques.

3. **`import sagemaker.amazon.common as smac`** :
   - Cette ligne importe un module spécifique de SageMaker, `sagemaker.amazon.common`, qui contient des fonctions utilitaires, notamment `write_numpy_to_dense_tensor`, pour convertir des données `numpy` en un format dense compatible avec les algorithmes SageMaker.

```python
train_set_vectors = np.array([t.tolist() for t in train_set[0]]).astype("float32")
```

4. **`train_set_vectors = np.array([t.tolist() for t in train_set[0]]).astype("float32")`** :
   - Cette ligne convertit les images d'entraînement de `train_set[0]` en un tableau `numpy`.
   - **`[t.tolist() for t in train_set[0]]`** : Pour chaque image `t` dans `train_set[0]`, la méthode `tolist()` est appliquée pour transformer l'image en une liste. Cela crée une liste de listes, où chaque sous-liste représente une image.
   - **`np.array(...).astype("float32")`** : La liste de listes est ensuite convertie en un tableau `numpy` de type `float32`, un format numérique à virgule flottante, nécessaire pour l'entraînement de modèles de machine learning.

```python
train_set_labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype("float32")
```

5. **`train_set_labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype("float32")`** :
   - Cette ligne prépare les étiquettes (labels) pour les images d'entraînement en les convertissant en tableau `numpy` avec des valeurs de type `float32`.
   - **`np.array([t.tolist() for t in train_set[1]])`** : Convertit les étiquettes en une liste de listes, puis en un tableau `numpy`.
   - **`np.where(... == 0, 1, 0)`** : Applique une transformation pour que toutes les étiquettes égales à 0 (ex : images de chiffre zéro) soient converties en 1, et toutes les autres étiquettes en 0. Ceci crée un problème de classification binaire (zéro vs non-zéro).
   - **`.astype("float32")`** : Le résultat est converti en type `float32`.

```python
validation_set_vectors = np.array([t.tolist() for t in valid_set[0]]).astype("float32")
validation_set_labels = np.where(np.array([t.tolist() for t in valid_set[1]]) == 0, 1, 0).astype("float32")
```

6. **`validation_set_vectors` et `validation_set_labels`** :
   - Ces lignes font la même chose que pour les données d'entraînement mais appliqué aux données de validation (`valid_set`). `validation_set_vectors` contient les vecteurs d’images de validation, et `validation_set_labels` contient les labels convertis en binaire (0 ou 1) pour le jeu de validation.

```python
train_set_buf = io.BytesIO()
validation_set_buf = io.BytesIO()
```

7. **`train_set_buf = io.BytesIO()` et `validation_set_buf = io.BytesIO()`** :
   - Ces lignes créent des objets `BytesIO` en mémoire, `train_set_buf` et `validation_set_buf`, qui serviront de tampons (buffers) pour stocker les données d'entraînement et de validation respectivement, en format binaire.
   - Ces tampons en mémoire sont pratiques pour écrire des données sous forme de flux binaire avant de les envoyer ou de les charger dans SageMaker.

```python
smac.write_numpy_to_dense_tensor(train_set_buf, train_set_vectors, train_set_labels)
smac.write_numpy_to_dense_tensor(validation_set_buf, validation_set_vectors, validation_set_labels)
```

8. **`smac.write_numpy_to_dense_tensor(...)`** :
   - Cette fonction de `sagemaker.amazon.common` écrit les données dans les tampons en mémoire au format "dense tensor", un format attendu par certains algorithmes de SageMaker.
   - **`train_set_buf` et `validation_set_buf`** : Ces tampons stockent les données au format requis pour SageMaker après l'appel à `write_numpy_to_dense_tensor`.
   - **`train_set_vectors` et `validation_set_vectors`** : Contiennent les vecteurs d'image sous forme de matrices `numpy`.
   - **`train_set_labels` et `validation_set_labels`** : Contiennent les étiquettes des données d’entraînement et de validation dans un format binaire (0 ou 1).

```python
train_set_buf.seek(0)
validation_set_buf.seek(0)
```

9. **`train_set_buf.seek(0)` et `validation_set_buf.seek(0)`** :
   - `seek(0)` repositionne le curseur au début de chaque tampon (`train_set_buf` et `validation_set_buf`). Cela permet de relire les données depuis le début du flux lorsque ces tampons sont transmis ou chargés dans un modèle.

En résumé, ce code prépare les données et les labels en tant que tableaux `numpy` au format attendu par SageMaker, les stocke dans des tampons binaires en mémoire (`BytesIO`), puis écrit les données en format dense tensor pour qu’elles puissent être facilement utilisées dans les algorithmes de machine learning de SageMaker.



-----------------------------
-----------------------------
-----------------------------
-----------------------------
-----------------------------


# 5. **Téléchargement vers S3**

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

------------
# Explication
------------

```python
key = "recordio-pb-data"
```

1. **`key = "recordio-pb-data"`** :
   - Cette ligne définit une clé, ou nom de fichier, sous forme de chaîne de caractères, `recordio-pb-data`, qui servira de nom pour les fichiers de données d'entraînement et de validation lorsqu'ils seront téléchargés dans le bucket S3. Ce nom est utilisé pour structurer et identifier les fichiers dans le bucket.

```python
boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "train", key)).upload_fileobj(train_set_buf)
```

2. **`boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "train", key)).upload_fileobj(train_set_buf)`** :
   - Cette ligne utilise le module `boto3` pour télécharger le contenu de `train_set_buf` (tampon de données d'entraînement) dans le bucket S3 spécifié.
   - **`boto3.resource("s3")`** : Crée une ressource S3 permettant d'interagir avec S3 en mode "resource", ce qui offre une interface orientée objet.
   - **`Bucket(bucket).Object(os.path.join(prefix, "train", key))`** : Sélectionne un objet spécifique dans le bucket. Le chemin de l’objet est construit avec `os.path.join(prefix, "train", key)`, qui crée un sous-dossier `train` sous le `prefix` dans le bucket, où le fichier sera stocké sous le nom défini par `key`.
   - **`upload_fileobj(train_set_buf)`** : Télécharge le contenu du tampon `train_set_buf` dans le chemin S3 défini. `train_set_buf` contient les données d’entraînement converties en format dense.

```python
boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "validation", key)).upload_fileobj(validation_set_buf)
```

3. **`boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, "validation", key)).upload_fileobj(validation_set_buf)`** :
   - Cette ligne est similaire à la précédente, mais elle télécharge le contenu de `validation_set_buf` (tampon de données de validation) dans un autre dossier du bucket S3.
   - **`os.path.join(prefix, "validation", key)`** : Utilise le même `key` et `prefix`, mais le fichier est stocké dans un sous-dossier `validation` au lieu de `train`.
   - **`upload_fileobj(validation_set_buf)`** : Télécharge le contenu du tampon `validation_set_buf` dans le chemin S3 spécifié.

```python
s3_train_data = f"s3://{bucket}/{prefix}/train/{key}"
```

4. **`s3_train_data = f"s3://{bucket}/{prefix}/train/{key}"`** :
   - Définit l’URL S3 pour les données d'entraînement sous forme de chaîne. Elle suit le format `s3://<nom_bucket>/<chemin>`.
   - Cette URL est sauvegardée dans la variable `s3_train_data`, ce qui permet d'accéder facilement aux données d’entraînement dans S3 depuis SageMaker ou d'autres services AWS.

```python
s3_validation_data = f"s3://{bucket}/{prefix}/validation/{key}"
```

5. **`s3_validation_data = f"s3://{bucket}/{prefix}/validation/{key}"`** :
   - Définit l’URL S3 pour les données de validation, en utilisant un chemin qui pointe vers le sous-dossier `validation` dans le bucket S3.
   - `s3_validation_data` permet de stocker l’URL pour y accéder facilement lors de l’entraînement d’un modèle.

```python
output_location = f"s3://{bucket}/{prefix}/output"
```

6. **`output_location = f"s3://{bucket}/{prefix}/output"`** :
   - Définit l'URL pour l’emplacement de sortie des résultats d’entraînement (comme les modèles ou métriques de performance). Les résultats seront enregistrés dans un sous-dossier `output` du même bucket S3.
   - Cette URL est souvent fournie aux modèles SageMaker pour spécifier où sauvegarder les résultats une fois l'entraînement terminé.

En résumé, ce code télécharge les données d'entraînement et de validation dans un bucket S3 en créant des URLs pour chaque fichier et définit également un emplacement pour stocker les résultats de l’entraînement. Ces URLs peuvent ensuite être utilisées dans SageMaker pour configurer et gérer le processus de formation du modèle.


-----------------------------
-----------------------------
-----------------------------
-----------------------------
-----------------------------


# Entraînement du Modèle Linéaire
-----------------------------------------

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


----------------------
# Annexe - explications finales
----------------------

### 1. **Configuration de l'entraînement**

Cette première section initialise le modèle SageMaker Linear Learner et configure les paramètres pour l'entraînement.

```python
from sagemaker import image_uris
```

1. **`from sagemaker import image_uris`** :
   - Importe la fonction `image_uris` de SageMaker. Cette fonction permet de récupérer l’URI du conteneur Docker contenant le modèle Linear Learner pour la région AWS spécifique, ce qui est nécessaire pour l’entraînement.

```python
container = image_uris.retrieve(region=boto3.Session().region_name, framework="linear-learner")
```

2. **`container = image_uris.retrieve(...)`** :
   - **`image_uris.retrieve(...)`** : Récupère l'URI du conteneur pour le modèle "linear-learner" dans la région AWS actuelle.
   - **`region=boto3.Session().region_name`** : Spécifie la région en utilisant la région par défaut définie dans la session `boto3`.
   - **`framework="linear-learner"`** : Indique le type de modèle à utiliser (ici, un modèle Linear Learner de SageMaker).

```python
linear = sagemaker.estimator.Estimator(
   container,
   role,
   instance_count=1,
   instance_type="ml.c4.xlarge",
   output_path=output_location,
   sagemaker_session=sess,
)
```

3. **`linear = sagemaker.estimator.Estimator(...)`** :
   - Crée un objet `Estimator` pour le modèle Linear Learner.
   - **`container`** : Définit le conteneur à utiliser, récupéré précédemment.
   - **`role`** : Spécifie le rôle IAM (Identity and Access Management) qui permet à SageMaker d’accéder aux ressources AWS nécessaires.
   - **`instance_count=1`** : Définit le nombre d'instances pour l'entraînement (ici, une seule instance).
   - **`instance_type="ml.c4.xlarge"`** : Spécifie le type d'instance pour l'entraînement, ici une instance `ml.c4.xlarge`.
   - **`output_path=output_location`** : Indique où les résultats d'entraînement seront stockés dans S3.
   - **`sagemaker_session=sess`** : Utilise la session SageMaker spécifiée pour gérer l’entraînement.

```python
linear.set_hyperparameters(feature_dim=784, predictor_type="binary_classifier", mini_batch_size=200)
```

4. **`linear.set_hyperparameters(...)`** :
   - Configure les hyperparamètres pour l'entraînement.
   - **`feature_dim=784`** : Définit la dimension de l'entrée (ici, 784, correspondant aux 28x28 pixels des images MNIST).
   - **`predictor_type="binary_classifier"`** : Spécifie le type de tâche, ici la classification binaire.
   - **`mini_batch_size=200`** : Définit la taille du mini-batch pour l'entraînement (nombre d'exemples traités en une fois).

```python
linear.fit({"train": s3_train_data})
```

5. **`linear.fit({"train": s3_train_data})`** :
   - Démarre l'entraînement du modèle avec les données fournies.
   - **`{"train": s3_train_data}`** : Spécifie que les données d'entraînement se trouvent dans `s3_train_data`, l’URL S3 vers le dataset.

---

### 2. **Optimisation Automatique des Hyperparamètres**

Cette section utilise la fonction de tuning des hyperparamètres pour optimiser le modèle.

```python
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
```

6. **`from sagemaker.tuner import ...`** :
   - Importe les classes `IntegerParameter`, `ContinuousParameter`, et `HyperparameterTuner` pour configurer le tuning automatique.

```python
hyperparameter_ranges = {
   "wd": ContinuousParameter(1e-7, 1),
   "learning_rate": ContinuousParameter(1e-5, 1),
   "mini_batch_size": IntegerParameter(100, 2000),
}
```

7. **`hyperparameter_ranges = {...}`** :
   - Définit la plage de valeurs pour chaque hyperparamètre à ajuster.
   - **`wd`** : Poids de régularisation (Regularization weight decay) avec une plage continue entre \(1 \times 10^{-7}\) et 1.
   - **`learning_rate`** : Taux d'apprentissage avec une plage continue entre \(1 \times 10^{-5}\) et 1.
   - **`mini_batch_size`** : Taille du mini-batch, définie ici comme un nombre entier entre 100 et 2000.

```python
hp_tuner = HyperparameterTuner(
   linear,
   "validation:binary_f_beta",
   hyperparameter_ranges,
   max_jobs=6,
   max_parallel_jobs=2,
   objective_type="Maximize",
)
```

8. **`hp_tuner = HyperparameterTuner(...)`** :
   - Crée un objet `HyperparameterTuner` pour gérer le tuning.
   - **`linear`** : Modèle Linear Learner à optimiser.
   - **`"validation:binary_f_beta"`** : Métrique cible (précision F-Beta sur les données de validation).
   - **`hyperparameter_ranges`** : Plages de valeurs pour les hyperparamètres.
   - **`max_jobs=6`** : Nombre maximal d'essais.
   - **`max_parallel_jobs=2`** : Nombre d'essais pouvant être exécutés simultanément.
   - **`objective_type="Maximize"`** : Cherche à maximiser la métrique.

```python
hp_tuner.fit(inputs={"train": s3_train_data, "validation": s3_validation_data})
```

9. **`hp_tuner.fit(...)`** :
   - Lance le processus de tuning en fournissant les données d'entraînement et de validation.

---

### 3. **Déploiement et Validation du Modèle**

#### 1. **Déploiement de l’Endpoint**

```python
linear_predictor = hp_tuner.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
```

10. **`linear_predictor = hp_tuner.deploy(...)`** :
    - Déploie le meilleur modèle trouvé par le tuner en créant un endpoint.
    - **`initial_instance_count=1`** : Nombre d'instances pour l'endpoint.
    - **`instance_type="ml.m4.xlarge"`** : Type d'instance pour le déploiement.

#### 2. **Validation des Prédictions**

```python
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
```

11. **`from sagemaker.serializers import CSVSerializer`** et **`from sagemaker.deserializers import JSONDeserializer`** :
    - Importe `CSVSerializer` pour sérialiser les données en format CSV et `JSONDeserializer` pour désérialiser les prédictions au format JSON.

```python
linear_predictor.serializer = CSVSerializer()
linear_predictor.deserializer = JSONDeserializer()
```

12. **`linear_predictor.serializer = CSVSerializer()`** et **`linear_predictor.deserializer = JSONDeserializer()`** :
    - Configure le prédicteur pour envoyer les données sous format CSV et recevoir les prédictions sous format JSON.

```python
result = linear_predictor.predict(train_set[0][30:31], initial_args={"ContentType": "text/csv"})
print(result)
```

13. **`result = linear_predictor.predict(...)`** :
    - Effectue une prédiction sur une seule observation (ici, l’image de train_set[0][30]).
    - **`initial_args={"ContentType": "text/csv"}`** : Spécifie que les données sont envoyées en format CSV.
    - **`print(result)`** : Affiche le résultat de la prédiction.

```python
predictions = []
for array in np.array_split(test_set[0], 100):
    result = linear_predictor.predict(array)
    predictions += [r["predicted_label"] for r in result["predictions"]]
```

14. **Prédictions en lot** :
    - **`np.array_split(test_set[0], 100)`** : Divise le jeu de test en 100 lots pour prédictions en série.
    - **`predictions += [r["predicted_label"] for r in result["predictions"]]`** : Récupère les prédictions de chaque lot et ajoute les labels prévus à `predictions`.

---

### 4. **Évaluation du Modèle**

```python
import pandas as pd
pd.crosstab(
    np.where(test_set[1] == 0, 1, 0),
    predictions,
    rownames=["réels"],
    colnames=["prédictions"]
)
```

15. **Matrice de confusion** :
    - Utilise `pandas.crosstab` pour générer une matrice de confusion, comparant les valeurs réelles et prédictions.
    - **`np.where(test_set[1] == 0, 1, 0)`** : Transforme les labels réels pour une classification binaire (0 ou 1).

---

### 5. **(Optionnel) Suppression de l'Endpoint**

```python
linear_predictor.delete_model()
linear_predictor.delete_endpoint()
```

16. **Suppression de l'endpoint et du modèle** :
    - **`linear_predictor.delete_model()`** : Supprime le modèle déployé.
    - **`linear_predictor.delete_endpoint()`** : Supprime l’endpoint pour libérer les ressources AWS.
