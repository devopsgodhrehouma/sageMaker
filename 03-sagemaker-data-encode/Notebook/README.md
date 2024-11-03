----------------------
# Notebook 3.3 - Cahier de l'Étudiant
----------------------

*Ce code vous permet d'importer, d'explorer et d'encoder les données d'un jeu de données automobile, en transformant des données catégoriques en valeurs numériques adaptées aux modèles d'apprentissage automatique.*

```python
# Importer les bibliothèques nécessaires
import pandas as pd

# Configurer l'affichage pour voir toutes les colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Charger les données et définir les noms de colonnes
url = "imports-85.csv"
col_names = [
    'symboling', 'normalized-losses', 'fuel-type', 'aspiration', 'num-of-doors', 
    'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 
    'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 
    'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
]
df_car = pd.read_csv(url, sep=',', names=col_names, na_values="?", header=None)

# Dimensions du jeu de données
df_car.shape

# Afficher les premières lignes
df_car.head(5)

# Informations sur les colonnes
df_car.info()

# Sélection des colonnes à encoder
df_car = df_car[['aspiration', 'num-of-doors', 'drive-wheels', 'num-of-cylinders']].copy()
df_car.head()
```

----------------------
# Importation des Bibliothèques et Chargement des Données
----------------------

Nous commençons par charger le jeu de données et configurer l'affichage pour visualiser toutes les colonnes dans le DataFrame.

```python
# Importer les bibliothèques nécessaires pour la manipulation des données
import pandas as pd

# Configurer l'affichage pour voir toutes les colonnes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Charger le fichier CSV et définir les noms des colonnes
url = "imports-85.csv"
col_names = [
    'symboling', 'normalized-losses', 'fuel-type', 'aspiration', 'num-of-doors', 
    'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 
    'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 
    'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
]
df_car = pd.read_csv(url, sep=',', names=col_names, na_values="?", header=None)
```

----------------------
# Exploration des Données
----------------------

Une fois les données chargées, nous examinons leurs dimensions, les premières lignes, et les informations générales sur les colonnes.

```python
# Dimensions du jeu de données
df_car.shape

# Afficher les premières lignes
df_car.head(5)

# Informations sur les colonnes
df_car.info()
```

Nous simplifions ensuite le jeu de données en sélectionnant les colonnes nécessaires pour l'encodage.

```python
# Sélection des colonnes à encoder
df_car = df_car[['aspiration', 'num-of-doors', 'drive-wheels', 'num-of-cylinders']].copy()
df_car.head()
```

----------------------
# Étape 1 : Encodage des Données Catégoriques Ordinales
----------------------

Nous commençons par encoder les colonnes `num-of-doors` et `num-of-cylinders`, qui sont des variables ordinales avec un ordre défini entre leurs valeurs.

### Encodage de `num-of-doors`

```python
# Compter les valeurs uniques dans 'num-of-doors'
df_car['num-of-doors'].value_counts()

# Mapper les valeurs de 'num-of-doors'
door_mapper = {"two": 2, "four": 4}
df_car['doors'] = df_car["num-of-doors"].replace(door_mapper)

# Afficher le DataFrame pour vérifier le changement
df_car.head()
```

### Encodage de `num-of-cylinders`

De manière similaire, nous créons un dictionnaire pour mapper les valeurs de `num-of-cylinders` à leurs équivalents numériques.

```python
# Compter les valeurs uniques dans 'num-of-cylinders'
df_car['num-of-cylinders'].value_counts()

# Mapper les valeurs de 'num-of-cylinders'
cylinder_mapper = {
    "two": 2, "three": 3, "four": 4, "five": 5, 
    "six": 6, "eight": 8, "twelve": 12
}
df_car['cylinders'] = df_car['num-of-cylinders'].replace(cylinder_mapper)

# Vérifier les nouvelles colonnes
df_car.head()
```

----------------------
# Étape 2 : Encodage des Données Catégoriques Non Ordinales
----------------------

Pour les colonnes `aspiration` et `drive-wheels`, nous appliquons **one-hot encoding** (ou encodage binaire) car elles ne possèdent pas d'ordre naturel entre leurs valeurs.

### Encodage de `drive-wheels`

```python
# Compter les valeurs uniques dans 'drive-wheels'
df_car['drive-wheels'].value_counts()

# Encodage binaire avec get_dummies
df_car = pd.get_dummies(df_car, columns=['drive-wheels'])

# Afficher le DataFrame avec les nouvelles colonnes
df_car.head()
```

L'encodage crée trois nouvelles colonnes `drive-wheels_4wd`, `drive-wheels_fwd`, et `drive-wheels_rwd`, chacune indiquant la présence (1) ou l'absence (0) d'une valeur.

### Encodage de `aspiration`

Pour la colonne `aspiration`, qui ne contient que deux valeurs (`std` et `turbo`), nous pouvons utiliser l'option `drop_first=True` pour éviter une redondance de colonnes.

```python
# Compter les valeurs uniques dans 'aspiration'
df_car['aspiration'].value_counts()

# Encodage binaire avec drop_first pour aspiration
df_car = pd.get_dummies(df_car, columns=['aspiration'], drop_first=True)

# Afficher le DataFrame final
df_car.head()
```

----------------------
# Conclusion
----------------------

Ce laboratoire vous a permis d'importer, d'explorer et de transformer un jeu de données automobile en encodant des caractéristiques ordinales et non ordinales pour les adapter aux modèles de machine learning.

**Défi** : Revenez au début du laboratoire et ajoutez d'autres colonnes du jeu de données. Expérimentez différentes méthodes d'encodage pour adapter chaque colonne selon sa nature ordinale ou non ordinale.

---
# Annexe
---


