
# Cahier de l'Étudiant

## Aperçu

Ce laboratoire est une continuation des exercices guidés du Module 3. Vous allez analyser un jeu de données biomédical pour prédire si un patient présente une anomalie orthopédique en fonction de six caractéristiques biomécaniques.

### Scénario d'Affaires

Vous travaillez pour un fournisseur de soins de santé, et vous avez pour mission d'améliorer la détection des anomalies chez les patients orthopédiques à l'aide de l'apprentissage automatique. Le jeu de données contient des caractéristiques biomécaniques et un label de classification (normal ou anormal) pour chaque patient.

### À propos du Jeu de Données

Ce jeu de données biomédical a été conçu par le Dr Henrique da Mota au Centre Médico-Chirurgical de Réadaptation des Massues à Lyon, en France. Il propose deux tâches de classification :
1. **Première tâche** : Classification des patients en trois catégories :
   - Normal (100 patients)
   - Hernie discale (60 patients)
   - Spondylolisthésis (150 patients)
2. **Deuxième tâche** : Fusionner les catégories Hernie discale et Spondylolisthésis en une catégorie "anormal". Il reste alors deux catégories :
   - Normal (100 patients)
   - Anormal (210 patients)

### Description des Attributs

Chaque patient est représenté par six attributs biomécaniques liés à la forme et à l’orientation du bassin et de la colonne lombaire :

1. Incidence pelvienne
2. Inclinaison pelvienne
3. Angle de lordose lombaire
4. Inclinaison sacrée
5. Rayon pelvien
6. Degré de spondylolisthésis

Les labels de classe sont définis comme suit :
- DH : Hernie discale
- SL : Spondylolisthésis
- NO : Normal
- AB : Anormal

Pour plus d'informations sur ce jeu de données, vous pouvez consulter la [page du jeu de données Vertebral Column](http://archive.ics.uci.edu/ml).

---

## Installation et Chargement des Données

Pour ce laboratoire, suivez les étapes ci-dessous pour configurer l'environnement et charger les données.

### Importation des Données

```python
import warnings, requests, zipfile, io
import pandas as pd
from scipy.io import arff

warnings.simplefilter('ignore')

# Téléchargement du fichier
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip, stream=True)
Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
Vertebral_zip.extractall()

# Chargement du fichier .arff
data = arff.loadarff('column_2C_weka.arff')
df = pd.DataFrame(data[0])
```

---

## Étape 1 : Exploration des Données

### 1.1 Examiner la Taille des Données

Obtenez le nombre de lignes et de colonnes.

```python
df.shape
```

### 1.2 Afficher les Colonnes

Obtenez la liste des colonnes, qui contient les six caractéristiques biomécaniques et la colonne de classe.

```python
df.columns
```

### 1.3 Types des Colonnes

Observez les types de données.

```python
df.dtypes
```

### 1.4 Statistiques Descriptives

Utilisez `describe` pour examiner les statistiques des colonnes, comme l'incidence pelvienne :

```python
df['pelvic_incidence'].describe()
```

**Défi :** Modifiez le code pour afficher les statistiques d'autres caractéristiques. Recherchez les valeurs extrêmes que vous pourriez souhaiter examiner de plus près.

---

## Étape 2 : Visualisation des Données

Pour mieux comprendre les données, utilisez des graphiques pour observer les distributions.

### 2.1 Visualisation des Valeurs

Utilisez un graphique pour visualiser chaque caractéristique.

```python
import matplotlib.pyplot as plt
%matplotlib inline

df.plot()
plt.show()
```

### 2.2 Graphiques de Densité

Visualisez les distributions des valeurs pour chaque caractéristique avec un graphique de densité (KDE).

```python
df.plot(kind='density', subplots=True, layout=(4,2), figsize=(12,12), sharex=False)
plt.show()
```

---

## Étape 3 : Analyse du Degré de Spondylolisthésis

Examinez les valeurs de `degree_spondylolisthesis` :

1. **Densité :** Montrez la distribution des valeurs.

   ```python
   df['degree_spondylolisthesis'].plot.density()
   ```

2. **Histogramme :** Observez la fréquence des valeurs.

   ```python
   df['degree_spondylolisthesis'].plot.hist()
   ```

3. **Boîte à Moustaches (Box Plot) :** Repérez les valeurs extrêmes éventuelles.

   ```python
   df['degree_spondylolisthesis'].plot.box()
   ```

---

## Étape 4 : Analyse de la Cible

1. **Distribution de la Cible**

   Comptez le nombre de valeurs pour chaque classe.

   ```python
   df['class'].value_counts()
   ```

2. **Conversion de la Cible en Valeurs Numériques**

   Modifiez la colonne `class` pour une utilisation avec le modèle.

   ```python
   class_mapper = {b'Abnormal':1, b'Normal':0}
   df['class'] = df['class'].replace(class_mapper)
   ```

3. **Corrélation avec `degree_spondylolisthesis`**

   Tracez un nuage de points pour visualiser la relation avec la cible.

   ```python
   df.plot.scatter(y='degree_spondylolisthesis', x='class')
   ```

**Défi :** Utilisez les cellules précédentes pour observer comment d'autres caractéristiques se comportent vis-à-vis de la cible.

---

## Étape 5 : Visualisation de Multiples Variables

Les visualisations peuvent aider à repérer des différences entre les valeurs normales et anormales.

### 5.1 Boîtes à Moustaches Groupées

Utilisez `groupby` pour afficher les caractéristiques pour chaque classe.

```python
df.groupby('class').boxplot(fontsize=20, rot=90, figsize=(20,10), patch_artist=True)
```

### 5.2 Matrice de Corrélation

Créez une matrice de corrélation pour observer les relations entre les caractéristiques.

```python
corr_matrix = df.corr()
corr_matrix["class"].sort_values(ascending=False)
```

### 5.3 Matrice de Dispersion

Affichez une matrice de dispersion pour observer les corrélations visuellement.

```python
pd.plotting.scatter_matrix(df, figsize=(12,12))
plt.show()
```

### 5.4 Carte de Chaleur (Heatmap)

Utilisez Seaborn pour visualiser la matrice de corrélation en utilisant une carte de chaleur.

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.color_palette("BrBG", 10)
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f")
plt.show()
```

**Défi :** Trouvez un autre jeu de données sur le [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml) et explorez-le en vous inspirant du code ci-dessus !

---

## Félicitations !

Vous avez terminé ce laboratoire. Suivez les instructions du guide pour clôturer la session.
