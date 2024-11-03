----------------------
# Notebook 2
----------------------

*Ce code vous permet d'importer, d'explorer et de visualiser les données du jeu de données biomédical sur les patients orthopédiques.*

```python
# Importer les bibliothèques nécessaires
import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff

warnings.simplefilter('ignore')

# Téléchargement et extraction des données
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Chargement du fichier .arff
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])

# Examiner la taille des données
df.shape

# Obtenir la liste des colonnes
df.columns

# Afficher les types de colonnes
df.dtypes

# Statistiques descriptives pour une colonne
df['pelvic_incidence'].describe()

# Statistiques descriptives pour l'ensemble du DataFrame
df.describe()

# Importer matplotlib pour la visualisation
import matplotlib.pyplot as plt
%matplotlib inline

# Visualiser les valeurs de chaque caractéristique
df.plot()
plt.show()

# Graphiques de densité
df.plot(kind='density', subplots=True, layout=(4,2), figsize=(12,12), sharex=False)
plt.show()

# Densité pour la colonne degree_spondylolisthesis
df['degree_spondylolisthesis'].plot.density()

# Histogramme pour degree_spondylolisthesis
df['degree_spondylolisthesis'].plot.hist()

# Boîte à moustaches pour degree_spondylolisthesis
df['degree_spondylolisthesis'].plot.box()

# Distribution de la cible
df['class'].value_counts()

# Conversion de la cible en valeurs numériques
class_mapper = {b'Abnormal':1, b'Normal':0}
df['class'] = df['class'].replace(class_mapper)

# Nuage de points pour degree_spondylolisthesis contre la cible
df.plot.scatter(y='degree_spondylolisthesis', x='class')

# Boîtes à moustaches groupées par classe
df.groupby('class').boxplot(fontsize=20, rot=90, figsize=(20,10), patch_artist=True)

# Matrice de corrélation
corr_matrix = df.corr()
corr_matrix["class"].sort_values(ascending=False)

# Matrice de dispersion
pd.plotting.scatter_matrix(df, figsize=(12,12))
plt.show()

# Carte de chaleur pour la matrice de corrélation
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.color_palette("BrBG", 10)
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")
plt.show()
``` 

----------------------
# Importation des Bibliothèques et Chargement des Données
----------------------

Nous commençons par importer les bibliothèques nécessaires et charger le fichier de données dans notre environnement de travail.

```python
# Importer les bibliothèques nécessaires pour gérer les avertissements, télécharger les données, et lire le format de fichier ARFF
import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff

# Ignorer les avertissements pour avoir un environnement de travail plus propre
warnings.simplefilter('ignore')

# Définir l'URL du fichier compressé contenant le jeu de données
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'

# Envoyer une requête HTTP pour télécharger le fichier
r = requests.get(f_zip, stream=True)

# Ouvrir le fichier zip en mémoire et extraire son contenu
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Charger le fichier .arff extrait dans un DataFrame Pandas
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])
```

----------------------
# Explication du Code
----------------------

1. **warnings** : La bibliothèque `warnings` nous permet de gérer les avertissements, en les désactivant ici pour simplifier la sortie.
2. **requests** : `requests.get()` télécharge le fichier compressé à partir de l'URL fournie.
3. **zipfile** : `zipfile.ZipFile()` est utilisé pour manipuler des fichiers `.zip`. Ici, il extrait le contenu dans le répertoire de travail.
4. **arff** : Le format `.arff` est un format de fichier utilisé pour les données de la base de données UCI. `arff.loadarff()` le lit et retourne un tableau de données, que nous convertissons ensuite en un DataFrame Pandas pour faciliter l'analyse.

----------------------
# Exploration des Données
----------------------

Une fois les données chargées, nous commençons par explorer leurs dimensions, leurs colonnes et leurs types pour bien comprendre ce que chaque colonne représente.

### Taille des Données
```python
# Examiner le nombre de lignes et de colonnes dans le jeu de données
df.shape
```
- **df.shape** : Affiche le nombre de lignes et de colonnes dans le DataFrame pour savoir combien d’enregistrements (patients) et de caractéristiques sont présents dans le jeu de données.

### Liste des Colonnes
```python
# Obtenir la liste des noms de colonnes
df.columns
```
- **df.columns** : Affiche les noms des colonnes, qui incluent les six caractéristiques biomécaniques et la colonne cible (`class`).

### Types des Colonnes
```python
# Afficher les types de données pour chaque colonne
df.dtypes
```
- **df.dtypes** : Montre les types de données pour chaque colonne. Ici, nous devrions voir que les six caractéristiques biomécaniques sont des `floats`, tandis que `class` est un type catégorique.

### Statistiques Descriptives

Nous allons utiliser la fonction `describe()` pour obtenir des statistiques descriptives sur les colonnes.

```python
# Obtenir les statistiques de la colonne 'pelvic_incidence'
df['pelvic_incidence'].describe()
```
- **df['pelvic_incidence'].describe()** : Affiche des statistiques comme la moyenne, l'écart-type, le minimum, et les valeurs quartiles pour la colonne `pelvic_incidence`.

**Défi :** Utilisez `describe()` sur d'autres colonnes pour rechercher des valeurs extrêmes.

### Statistiques Descriptives pour Tout le DataFrame

```python
# Obtenir les statistiques descriptives pour toutes les colonnes
df.describe()
```
- **df.describe()** : Résume les statistiques pour toutes les colonnes numériques du DataFrame. 

----------------------
# Visualisation des Données
----------------------

Nous allons maintenant visualiser les données pour observer les distributions et les éventuelles anomalies.

### Graphique des Caractéristiques

```python
import matplotlib.pyplot as plt
%matplotlib inline

# Tracer un graphique pour les six caractéristiques
df.plot()
plt.show()
```

- **df.plot()** : Génère un graphique de ligne pour chaque colonne du DataFrame afin de voir l'évolution des valeurs.

### Graphiques de Densité

Les graphiques de densité montrent la distribution des valeurs pour chaque caractéristique.

```python
# Tracer les graphiques de densité pour chaque caractéristique
df.plot(kind='density', subplots=True, layout=(4,2), figsize=(12,12), sharex=False)
plt.show()
```
- **kind='density'** : Spécifie un graphique de densité (KDE) pour chaque colonne.
- **subplots=True, layout=(4,2)** : Crée des sous-graphiques organisés en une grille de 4x2.
- **figsize=(12,12)** : Définit la taille de la figure.

----------------------
# Analyse du Degré de Spondylolisthésis
----------------------

Nous analysons la colonne `degree_spondylolisthesis` pour repérer des anomalies dans les données.

### Graphique de Densité

```python
# Densité pour 'degree_spondylolisthesis'
df['degree_spondylolisthesis'].plot.density()
```
- **plot.density()** : Affiche la distribution de `degree_spondylolisthesis`.

### Histogramme

```python
# Histogramme pour 'degree_spondylolisthesis'
df['degree_spondylolisthesis'].plot.hist()
```
- **plot.hist()** : Montre la fréquence des valeurs pour identifier les pics dans la distribution.

### Box Plot

```python
# Boîte à moustaches pour 'degree_spondylolisthesis'
df['degree_spondylolisthesis'].plot.box()
```
- **plot.box()** : Montre les valeurs extrêmes, les quartiles et les médianes.

----------------------
# Analyse de la Cible
----------------------

Pour entraîner un modèle de machine learning, nous devons examiner et transformer la colonne cible.

### Distribution de la Cible

```python
# Compter le nombre d'occurrences de chaque classe
df['class'].value_counts()
```
- **value_counts()** : Affiche le nombre d'occurrences pour chaque classe (`Normal` et `Abnormal`).

### Conversion de la Cible en Vale

urs Numériques

Les modèles de ML nécessitent des valeurs numériques pour la cible.

```python
# Mapper les classes en valeurs numériques
class_mapper = {b'Abnormal':1, b'Normal':0}
df['class'] = df['class'].replace(class_mapper)
```
- **replace(class_mapper)** : Remplace les valeurs `Normal` et `Abnormal` par 0 et 1 respectivement.

### Scatter Plot pour la Cible

Nous observons la relation entre `degree_spondylolisthesis` et `class`.

```python
# Scatter plot pour observer la relation entre la classe et le degré de spondylolisthésis
df.plot.scatter(y='degree_spondylolisthesis', x='class')
```
- **plot.scatter()** : Affiche la répartition des valeurs de `degree_spondylolisthesis` par rapport aux classes `Normal` et `Abnormal`.

----------------------
# Visualisation de Multiples Variables
----------------------

### Box Plot Groupé par Classe

```python
# Boîtes à moustaches groupées par classe
df.groupby('class').boxplot(fontsize=20, rot=90, figsize=(20,10), patch_artist=True)
```
- **groupby('class')** : Regroupe les données par classe.
- **boxplot()** : Crée un box plot pour chaque caractéristique, séparé par classe.

### Matrice de Corrélation

La corrélation entre les variables peut nous donner des indications sur les relations entre les caractéristiques.

```python
# Créer une matrice de corrélation
corr_matrix = df.corr()
corr_matrix["class"].sort_values(ascending=False)
```
- **df.corr()** : Calcule la corrélation entre chaque paire de caractéristiques.
- **sort_values()** : Trie les corrélations pour observer les caractéristiques les plus corrélées avec `class`.

### Matrice de Dispersion

La matrice de dispersion montre les relations pairées entre les caractéristiques.

```python
# Matrice de dispersion
pd.plotting.scatter_matrix(df, figsize=(12,12))
plt.show()
```
- **scatter_matrix()** : Crée une matrice de graphiques de dispersion pour visualiser les relations entre les caractéristiques.

----------------------
# Carte de Chaleur
----------------------

Une carte de chaleur montre la corrélation entre les caractéristiques sous forme de couleurs.

```python
import seaborn as sns

# Créer une carte de chaleur pour la matrice de corrélation
fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.color_palette("BrBG", 10)
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")
plt.show()
```

- **sns.heatmap()** : Crée une carte de chaleur annotée pour observer les corrélations visuellement.
- **colormap** : Définit une palette de couleurs pour la carte.

---

Ce laboratoire vous a permis de charger, explorer, et analyser un jeu de données biomédical en utilisant des techniques d’analyse de données exploratoires (EDA).
