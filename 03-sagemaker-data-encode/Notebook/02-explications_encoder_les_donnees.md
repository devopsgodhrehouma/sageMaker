-------------------------------------
# Notebook 3
-----------------------------------


### Étapes:

1. Charger le jeu de données et configurer les paramètres d'affichage.
2. Explorer les dimensions et les colonnes du jeu de données.
3. Encoder les caractéristiques catégoriques ordinales (`num-of-doors` et `num-of-cylinders`) en valeurs numériques.
4. Encoder les caractéristiques catégoriques non ordinales (`drive-wheels` et `aspiration`) en utilisant l'encodage binaire (*one-hot encoding*).

Ces étapes préparent les données pour un modèle de machine learning en transformant toutes les valeurs catégoriques en valeurs numériques exploitables.

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

# Compter les valeurs uniques dans 'num-of-doors'
df_car['num-of-doors'].value_counts()

# Mapper les valeurs de 'num-of-doors'
door_mapper = {"two": 2, "four": 4}
df_car['doors'] = df_car["num-of-doors"].replace(door_mapper)

# Afficher le DataFrame pour vérifier le changement
df_car.head()

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

# Compter les valeurs uniques dans 'drive-wheels'
df_car['drive-wheels'].value_counts()

# Encodage binaire avec get_dummies
df_car = pd.get_dummies(df_car, columns=['drive-wheels'])

# Afficher le DataFrame avec les nouvelles colonnes
df_car.head()

# Compter les valeurs uniques dans 'aspiration'
df_car['aspiration'].value_counts()

# Encodage binaire avec drop_first pour aspiration
df_car = pd.get_dummies(df_car, columns=['aspiration'], drop_first=True)

# Afficher le DataFrame final
df_car.head()
```

-------------------
# Explication
-------------------

*Ces étapes transforment les données brutes en un format exploitable par des modèles d’apprentissage automatique, en convertissant toutes les valeurs catégoriques en valeurs numériques.*

### Étape 1 : Charger le jeu de données et configurer les paramètres d'affichage

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
```

1. **Importer `pandas`** : La bibliothèque `pandas` est essentielle pour la manipulation de données sous forme de DataFrames.
   
2. **Configurer l'affichage** :
   - `pd.set_option('display.max_rows', 500)`, `pd.set_option('display.max_columns', 500)`, et `pd.set_option('display.width', 1000)` permettent de visualiser jusqu'à 500 lignes et 500 colonnes du DataFrame, et augmentent la largeur pour afficher les données sans retour à la ligne.
   
3. **Charger les données** :
   - `pd.read_csv()` charge le fichier CSV depuis l'URL fournie. Les colonnes sont nommées selon la liste `col_names` et `na_values="?"` traite les valeurs `?` comme des valeurs manquantes (NaN). `header=None` indique qu'il n'y a pas de ligne d'en-tête dans le fichier d'origine.

### Étape 2 : Explorer les dimensions et les colonnes du jeu de données

```python
# Dimensions du jeu de données
df_car.shape

# Afficher les premières lignes
df_car.head(5)

# Informations sur les colonnes
df_car.info()
```

1. **Dimensions du jeu de données** :
   - `df_car.shape` retourne un tuple (n, m) indiquant le nombre de lignes (n) et de colonnes (m) du DataFrame.
   
2. **Premières lignes du DataFrame** :
   - `df_car.head(5)` affiche les cinq premières lignes, ce qui aide à visualiser un échantillon des données.
   
3. **Informations sur les colonnes** :
   - `df_car.info()` donne un résumé des colonnes, y compris le type de données de chaque colonne et le nombre de valeurs non nulles.

### Étape 3 : Sélection des colonnes à encoder

```python
# Sélection des colonnes à encoder
df_car = df_car[['aspiration', 'num-of-doors', 'drive-wheels', 'num-of-cylinders']].copy()
df_car.head()
```

1. **Sélection de colonnes** :
   - On sélectionne uniquement les colonnes que l’on souhaite encoder : `aspiration`, `num-of-doors`, `drive-wheels` et `num-of-cylinders`.
   - `copy()` crée une copie indépendante des données sélectionnées.

2. **Affichage des premières lignes** :
   - `df_car.head()` permet de vérifier que seules les colonnes choisies sont présentes dans le nouveau DataFrame.

### Étape 4 : Encodage des caractéristiques catégoriques ordinales

Les colonnes `num-of-doors` et `num-of-cylinders` contiennent des valeurs catégoriques ordinales, c’est-à-dire qu’il existe un ordre logique entre les valeurs (ex., `two` < `four` pour `num-of-doors`).

#### Encodage de `num-of-doors`

```python
# Compter les valeurs uniques dans 'num-of-doors'
df_car['num-of-doors'].value_counts()

# Mapper les valeurs de 'num-of-doors'
door_mapper = {"two": 2, "four": 4}
df_car['doors'] = df_car["num-of-doors"].replace(door_mapper)

# Afficher le DataFrame pour vérifier le changement
df_car.head()
```

1. **Comptage des valeurs** :
   - `value_counts()` affiche le nombre d'occurrences de chaque valeur unique dans `num-of-doors`.

2. **Mapping des valeurs** :
   - `door_mapper = {"two": 2, "four": 4}` crée un dictionnaire de correspondance pour transformer `two` en 2 et `four` en 4.
   - `df_car['num-of-doors'].replace(door_mapper)` applique cette transformation et crée une nouvelle colonne `doors` contenant les valeurs numériques.

3. **Vérification du DataFrame** :
   - `df_car.head()` permet de visualiser la nouvelle colonne `doors` avec les valeurs encodées.

#### Encodage de `num-of-cylinders`

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

1. **Comptage des valeurs uniques** :
   - `value_counts()` indique les différentes valeurs de `num-of-cylinders`.

2. **Mapping des valeurs** :
   - `cylinder_mapper` convertit chaque valeur en son équivalent numérique, par exemple `two` devient 2, `eight` devient 8.
   - La colonne encodée est stockée dans `cylinders`.

3. **Vérification du DataFrame** :
   - `df_car.head()` affiche la nouvelle colonne `cylinders` avec les valeurs numériques encodées.

### Étape 5 : Encodage des caractéristiques catégoriques non ordinales

Les colonnes `drive-wheels` et `aspiration` sont non ordinales, donc il n'y a pas de relation d'ordre entre les valeurs. On utilise l’encodage binaire (one-hot encoding).

#### Encodage de `drive-wheels`

```python
# Compter les valeurs uniques dans 'drive-wheels'
df_car['drive-wheels'].value_counts()

# Encodage binaire avec get_dummies
df_car = pd.get_dummies(df_car, columns=['drive-wheels'])

# Afficher le DataFrame avec les nouvelles colonnes
df_car.head()
```

1. **Comptage des valeurs uniques** :
   - `value_counts()` affiche les valeurs uniques de `drive-wheels` (`4wd`, `fwd`, `rwd`).

2. **One-Hot Encoding** :
   - `pd.get_dummies(df_car, columns=['drive-wheels'])` crée une colonne binaire pour chaque valeur unique dans `drive-wheels`. Par exemple, si la valeur est `4wd`, alors `drive-wheels_4wd` prendra la valeur 1, et les autres colonnes associées à `drive-wheels` auront la valeur 0.

3. **Vérification du DataFrame** :
   - `df_car.head()` permet de voir les nouvelles colonnes créées par l’encodage.

#### Encodage de `aspiration`

```python
# Compter les valeurs uniques dans 'aspiration'
df_car['aspiration'].value_counts()

# Encodage binaire avec drop_first pour aspiration
df_car = pd.get_dummies(df_car, columns=['aspiration'], drop_first=True)

# Afficher le DataFrame final
df_car.head()
```

1. **Comptage des valeurs uniques** :
   - `value_counts()` montre les valeurs `std` et `turbo` dans `aspiration`.

2. **One-Hot Encoding avec `drop_first`** :
   - `pd.get_dummies(df_car, columns=['aspiration'], drop_first=True)` encode `aspiration` en une colonne binaire (`aspiration_turbo`). En utilisant `drop_first=True`, `std` est implicite lorsque `aspiration_turbo` est 0.

3. **Vérification du DataFrame final** :
   - `df_car.head()` montre le DataFrame final avec toutes les colonnes encodées et prêtes pour l’analyse ou l’entraînement du modèle.



