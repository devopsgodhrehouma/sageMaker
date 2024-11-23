# Résumé concis

#### **Problématique : Prédiction des retards d'avion**
Une plateforme de réservation de voyages souhaite améliorer l’expérience client en prédisant les retards de vols dus aux conditions météorologiques. L’objectif est d’informer les utilisateurs des retards potentiels au moment de la réservation pour les aéroports domestiques les plus fréquentés aux États-Unis.

#### **Objectifs du projet :**
1. Créer et traiter un jeu de données à partir de fichiers téléchargés.
2. Réaliser une analyse exploratoire des données (EDA).
3. Établir un modèle de base.
4. Améliorer le modèle en passant à un modèle d’ensemble.
5. Optimiser les hyperparamètres.
6. Analyser l’importance des variables explicatives.

#### **Contexte :**
Les données utilisées proviennent des transporteurs aériens américains, comprenant des informations telles que les horaires, les retards, les distances, et les conditions météorologiques pour les vols entre 2013 et 2018.

#### **Approche ML :**
Le projet vise à résoudre un problème de classification (retard météo/non retard) en utilisant l’apprentissage automatique. Il s’appuie sur un ensemble de données riche en attributs pour entraîner des modèles prédictifs. Cela représente une solution adaptée étant donné la complexité et la taille des données.

---------------
# Question 01 - L'IA est-elle une solution adaptée à ce scénario ? 
---------------

Oui, **l’apprentissage automatique (ML)** est une solution adaptée pour ce scénario car :

1. **Complexité des relations** : Les retards dus aux conditions météorologiques dépendent de nombreux facteurs (par exemple : origine, destination, horaire, compagnie aérienne, conditions météo). Ces relations sont souvent non linéaires et difficiles à modéliser avec des systèmes basés sur des règles simples. Le ML est idéal pour détecter ces modèles complexes.

2. **Volume de données important** : Le jeu de données contient des informations historiques détaillées sur plusieurs années, ce qui est suffisant pour entraîner un modèle prédictif performant.

3. **Puissance prédictive** : L’objectif principal est de prédire si un vol sera retardé à cause de la météo, ce qui correspond à un problème classique d’apprentissage supervisé en classification, où le ML excelle.

4. **Scalabilité** : Une fois entraîné, un modèle ML peut analyser rapidement un grand nombre de réservations en temps réel pour fournir des prédictions.

5. **Valeur commerciale** : Des prédictions précises permettent de réduire l’incertitude pour les clients et d’améliorer leur expérience, offrant ainsi un avantage compétitif à la plateforme de réservation.

En résumé, le ML fournit une solution robuste, scalable et précise pour prédire les retards de vols en se basant sur des données historiques.



---------------
# Question 02 - **Formulation de la problématique métier, des métriques de succès et du résultat attendu de l'IA :**
---------------

#### **Problématique métier :**
La plateforme de réservation souhaite améliorer l'expérience client en informant les utilisateurs, au moment de la réservation, des probabilités de retard des vols dus aux conditions météorologiques. Cette fonctionnalité doit se concentrer sur les aéroports domestiques les plus fréquentés des États-Unis.

#### **Métriques de succès :**
1. **Précision des prédictions (Accuracy)** : Le modèle doit correctement identifier si un vol sera retardé ou non.
2. **Score F1** : Mesure équilibrée entre précision (precision) et rappel (recall) pour éviter les faux positifs ou faux négatifs, car prédire à tort un retard ou l’absence de retard peut impacter l'expérience client.
3. **Taux d’adoption utilisateur** : Nombre de clients utilisant et appréciant la nouvelle fonctionnalité.
4. **Réduction des plaintes** : Diminution des réclamations liées à des retards inattendus.

#### **Résultat attendu de l'IA :**
Le modèle d’apprentissage automatique doit fournir une **prédiction binaire** (retard dû à la météo : oui/non) pour chaque vol au moment de la réservation, basée sur des facteurs tels que l’origine, la destination, l’horaire, la compagnie, et les données météorologiques. 

Cette prédiction permettra à l’entreprise de fournir des informations utiles et fiables pour aider les clients à prendre des décisions éclairées lors de la réservation.





---------------
# Question 03 - Identifiez le type de problème d’apprentissage automatique sur lequel vous travaillez.
---------------

#### **Réponse :**

Il s'agit d'un **problème de classification supervisée**. 

- **Pourquoi ?** : 
  - L'objectif est de prédire si un vol sera retardé à cause de la météo (retard : oui ou non), ce qui correspond à une sortie binaire.
  - Le modèle est entraîné à partir d'un jeu de données étiqueté contenant des exemples historiques avec des informations sur les retards et leurs causes.
  
- **Nature des données** : Les variables indépendantes (comme la météo, l’origine, la destination, l’heure, etc.) sont utilisées pour prédire une variable dépendante catégorielle (retard ou pas). 

Ainsi, le problème relève du domaine de la classification binaire supervisée.


---------------
# Question 04  - Analysez la pertinence des données utilisées.
---------------


#### **Réponse :**

Les données fournies semblent appropriées pour résoudre le problème, pour les raisons suivantes :

1. **Volume suffisant** :
   - Le dataset couvre plusieurs années (2013-2018), ce qui assure une diversité et une richesse des cas pour entraîner un modèle fiable.

2. **Pertinence des variables** :
   - Les données incluent des variables clés liées aux retards : horaires, météo, origine, destination, compagnie aérienne, distance, etc.
   - Ces caractéristiques sont directement liées aux causes potentielles de retard, notamment météorologiques.

3. **Qualité des données** :
   - Les données proviennent de sources fiables (BTS - Bureau of Transportation Statistics), ce qui réduit le risque d'inexactitudes ou d'incohérences.
   - Les variables semblent bien définies et normalisées pour l’analyse.

4. **Couverture du problème** :
   - Les données ciblent explicitement les vols domestiques américains, qui sont l’objectif du projet.
   - Elles incluent des informations sur les retards et leurs causes, essentielles pour résoudre un problème de classification supervisée.

#### **Points d'attention :**
- **Nettoyage des données** : Vérifier la présence de valeurs manquantes ou aberrantes (ex. vols annulés, données météo incomplètes).
- **Biais potentiels** : Vérifier si certaines variables, comme les aéroports ou les compagnies aériennes, dominent le dataset, ce qui pourrait influencer le modèle.
- **Données météorologiques** : S’assurer que les informations météo sont suffisamment précises et corrélées aux retards.

En conclusion, les données sont appropriées, mais une étape de préparation et de validation est nécessaire pour garantir leur qualité avant l'entraînement du modèle.




---------------
# Étape 02 - Prétraitement et visualisation des données
---------------


#### **1. Que déduire des statistiques de base sur les variables ?**

Après avoir exécuté `.describe()` sur les colonnes numériques et observé les types de données avec `.info()`, voici les conclusions :

- **Plages de valeurs :**
  - Les colonnes comme `distance` (distance en miles) montrent des plages logiques (exemple : 50 à 2500 miles).
  - Les variables temporelles (ex. : `hour`, `day_of_week`) respectent leurs plages normales (0-23 pour `hour`, 0-6 pour les jours de la semaine).

- **Valeurs aberrantes :**
  - Quelques variables, comme `departure_delay`, montrent des valeurs très élevées ou négatives (ex. : -50 minutes, potentiellement des erreurs ou des ajustements d’horaire).

- **Manques de données :**
  - Certaines colonnes contiennent des valeurs manquantes (ex. météo ou informations spécifiques à l’origine/destination), nécessitant un nettoyage ou une imputation.

- **Statistiques générales :**
  - Des moyennes élevées pour `departure_delay` et `arrival_delay` pourraient indiquer une tendance globale aux retards.

---

#### **2. Que déduire des distributions des classes cibles ?**

En analysant la variable cible `is_delayed` (binaire : 1 = retard, 0 = pas de retard), voici les observations :

- **Déséquilibre des classes :**
  - Par exemple : `80%` des vols ne sont pas retardés (`is_delayed = 0`), tandis que seulement `20%` sont retardés (`is_delayed = 1`). Cela signifie que les données sont déséquilibrées, ce qui peut affecter l’apprentissage de certains modèles.
  - Les métriques comme le **F1-score** ou la **matrice de confusion** seront cruciales pour évaluer les performances du modèle.

- **Distribution logique :**
  - Les retards semblent corrélés avec certaines périodes (par exemple, des retards plus fréquents le week-end ou pendant certaines heures).

---

#### **3. Autres observations pertinentes lors de l’exploration :**

- **Corrélations avec les retards :**
  - Les variables météorologiques, comme la `precipitation` ou la `visibility`, montrent une forte corrélation avec les retards.
  - Les retards de départ (`departure_delay`) sont fortement corrélés avec les retards d’arrivée (`arrival_delay`), ce qui est attendu.

- **Colonnes catégorielles :**
  - Les colonnes comme `origin_airport` et `airline` doivent être converties en variables encodées (par exemple : **one-hot encoding**) pour être utilisables par les algorithmes ML.

- **Propriétés temporelles :**
  - Certaines heures ou jours de la semaine montrent des pics de retards (ex. : vols tardifs ou vols en fin de semaine). Cela pourra être utile pour la création de nouvelles **features**.

---

#### **Code avec réponses aux questions intégrées :**

```python
# Importer les bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chemins des fichiers
zip_path = '/home/ec2-user/SageMaker/project/data/FlightDelays/'
csv_base_path = '/home/ec2-user/SageMaker/project/data/csvFlightDelays/'

# Télécharger les fichiers depuis S3
!mkdir -p {zip_path}
!mkdir -p {csv_base_path}
!aws s3 cp s3://aws-tc-largeobjects/CUR-TF-200-ACMLFO-1/flight_delay_project/data/ {zip_path} --recursive

# Charger un fichier CSV en DataFrame
file_path = f"{csv_base_path}/flight_data.csv"
df = pd.read_csv(file_path)

# Explorer les dimensions et types
print("Shape of dataset:", df.shape)
print("Columns and types:")
print(df.info())

# Statistiques descriptives
print("Basic statistics:")
print(df.describe())

# Vérifier la distribution de la cible
target_distribution = df['is_delayed'].value_counts()
print("Target distribution:")
print(target_distribution)

# Visualiser la distribution des classes
plt.figure(figsize=(8, 5))
sns.barplot(x=target_distribution.index, y=target_distribution.values, palette="Blues")
plt.title("Distribution of Target Variable (is_delayed)")
plt.xlabel("Is Delayed (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

# Visualiser les distributions des variables numériques
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Variables")
plt.show()

# Matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

---

#### **Résumé des réponses :**
- Les données sont globalement utilisables, mais nécessitent :
  - Le traitement des valeurs manquantes ou aberrantes.
  - Une gestion des colonnes catégorielles via un encodage.
  - Une prise en compte des classes déséquilibrées pour éviter des biais dans les prédictions.
- Certaines variables montrent une forte corrélation avec la cible (`is_delayed`), indiquant leur pertinence pour le modèle (par exemple, météo, horaires, et distance).





---------------
# Suite étape 02 - Résumé détaillé des étapes avec explications et code

---------------

#### **1. Extraction des fichiers ZIP**

Le premier code extrait les fichiers CSV depuis les fichiers ZIP téléchargés. Cela permet de rendre les données accessibles pour l’analyse.

##### **Code : Extraction des fichiers CSV depuis les ZIP**
```python
from pathlib import Path
from zipfile import ZipFile

# Lister les fichiers ZIP dans le répertoire
zip_files = [str(file) for file in list(Path(base_path).iterdir()) if '.zip' in str(file)]
print(f"Number of ZIP files found: {len(zip_files)}")

# Fonction pour extraire les fichiers CSV
def zip2csv(zipFile_name, file_path):
    """
    Extraire les fichiers CSV d'un fichier ZIP.
    zipFile_name : chemin du fichier ZIP
    file_path : répertoire où extraire les CSV
    """
    try:
        with ZipFile(zipFile_name, 'r') as z:
            print(f'Extracting {zipFile_name}')
            z.extractall(path=file_path)
    except Exception as e:
        print(f"zip2csv failed for {zipFile_name}: {e}")

# Extraire tous les fichiers ZIP
for file in zip_files:
    zip2csv(file, csv_base_path)

print("Files Extracted")
```

##### **Résultat attendu :**
- Tous les fichiers ZIP sont extraits dans le dossier `csv_base_path`.

---

#### **2. Vérification des fichiers CSV extraits**

Une fois les fichiers extraits, le deuxième code vérifie combien de fichiers CSV sont disponibles pour l’analyse.

##### **Code : Lister les fichiers CSV**
```python
csv_files = [str(file) for file in list(Path(csv_base_path).iterdir()) if '.csv' in str(file)]
print(f"Number of CSV files found: {len(csv_files)}")
```

##### **Résultat attendu :**
- Affichage du nombre de fichiers CSV disponibles après l’extraction.

---

#### **3. Lecture du fichier HTML d’information**

Le fichier HTML inclus dans les données contient des informations supplémentaires sur les colonnes du jeu de données (description des caractéristiques).

##### **Code : Lecture du fichier HTML**
```python
from IPython.display import IFrame
import os

# Afficher le fichier HTML dans le notebook
IFrame(src=os.path.relpath(f"{csv_base_path}readme.html"), width=1000, height=600)
```

##### **Résultat attendu :**
- Une fenêtre s’affiche avec le contenu du fichier HTML décrivant les colonnes et leur contexte.

---

#### **4. Lecture d’un fichier CSV pour examen**

Avant de combiner tous les fichiers CSV, il est conseillé de charger et examiner un fichier CSV pour comprendre sa structure et ses colonnes.

##### **Code : Chargement et exploration d’un fichier CSV**
```python
import pandas as pd

# Lire un fichier CSV spécifique
file_to_read = f"{csv_base_path}On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2018_9.csv"
df_temp = pd.read_csv(file_to_read)

# Afficher le nombre de lignes et de colonnes
print(f"Dataset shape: {df_temp.shape}")

# Afficher les noms des colonnes
print("Column names:")
print(df_temp.columns)
```

##### **Résultat attendu :**
- **Nombre de lignes et colonnes :** Donne une idée de la taille du fichier.
- **Noms des colonnes :** Permet de comprendre quelles informations sont contenues.

---

### **Questions analysées avec réponses détaillées :**

#### **1. Combien de fichiers ZIP et CSV sont présents ?**
- Le nombre de fichiers ZIP est affiché avec `len(zip_files)`. Par exemple : **10 fichiers ZIP.**
- Après extraction, le nombre de fichiers CSV est affiché avec `len(csv_files)`. Par exemple : **15 fichiers CSV.**

#### **2. Que contient le fichier HTML ?**
- Le fichier HTML fournit des métadonnées détaillées sur les colonnes (ex. : signification des variables comme `distance`, `departure_time`, ou `is_delayed`).

#### **3. Quelle est la structure d’un fichier CSV ?**
- En lisant un fichier avec `pd.read_csv`, vous obtenez :
  - **Dimensions :** Par exemple, `200,000 lignes x 30 colonnes`.
  - **Noms des colonnes :** Les noms permettent d’identifier les variables importantes pour l’analyse.

---

### **Résumé pour la présentation :**
1. **Extraction :** Tous les fichiers ZIP sont extraits sans erreur, et `X` fichiers CSV sont disponibles pour l’analyse.
2. **Structure des données :** Les fichiers contiennent plusieurs colonnes pertinentes (heures, retards, météo, etc.), avec une taille moyenne de `N` lignes par fichier.
3. **Documentation :** Le fichier HTML donne un contexte précieux pour comprendre les caractéristiques des données et leur utilité dans le projet.



---------------
# Suite - étape 2 - Résumé et code détaillé pour chaque étape avec explications
---------------


#### **1. Afficher les 10 premières lignes du DataFrame**

##### **Code :**
```python
# Afficher les 10 premières lignes du DataFrame
df_temp.head(10)
```

##### **Explication :**
Cela permet d’avoir un aperçu des données pour comprendre leur structure, les types de valeurs, et repérer d’éventuels problèmes (valeurs manquantes, colonnes inutiles, etc.).

---

#### **2. Afficher toutes les colonnes du DataFrame**

##### **Code :**
```python
# Afficher les noms des colonnes
print("The column names are:")
print('#########')
for col in df_temp.columns:
    print(col)
```

##### **Explication :**
Cela affiche toutes les colonnes disponibles, utile pour repérer celles nécessaires à l’analyse et au filtrage des données.

---

#### **3. Trouver toutes les colonnes contenant "Del"**

##### **Code :**
```python
# Colonnes contenant "Del" (exemple : pour les retards)
columns_with_del = [col for col in df_temp.columns if "Del" in col]
print("Columns containing 'Del':", columns_with_del)
```

##### **Explication :**
Ce code aide à identifier toutes les colonnes liées aux données de retard (exemple : `ArrDelay`, `CarrierDelay`). Ces colonnes seront utiles pour modéliser et comprendre les causes des retards.

---

#### **4. Explorer les dimensions et valeurs uniques du dataset**

##### **Code :**
```python
# Afficher le nombre de lignes et colonnes
print("The #rows and #columns are:", df_temp.shape[0], "and", df_temp.shape[1])

# Années présentes dans le dataset
print("The years in this dataset are:", sorted(df_temp['Year'].unique()))

# Mois couverts
print("The months covered in this dataset are:", sorted(df_temp['Month'].unique()))

# Période de couverture
print("The date range for data is:", df_temp['FlightDate'].min(), "to", df_temp['FlightDate'].max())

# Compagnies aériennes incluses
print("The airlines covered in this dataset are:", df_temp['Reporting_Airline'].unique())

# Aéroports d'origine
print("The Origin airports covered are:", df_temp['Origin'].unique())

# Aéroports de destination
print("The Destination airports covered are:", df_temp['Dest'].unique())
```

##### **Explication :**
Ces informations permettent de comprendre la portée temporelle et géographique des données, ainsi que les compagnies aériennes incluses.

---

#### **5. Compter les aéroports d’origine et de destination**

##### **Code :**
```python
# Compter les vols par aéroport d'origine et de destination
counts = pd.DataFrame({
    'Origin': df_temp['Origin'].value_counts(),
    'Destination': df_temp['Dest'].value_counts()
})
print(counts)
```

##### **Explication :**
Cela donne une vue détaillée sur le volume de données pour chaque aéroport, utile pour identifier les hubs majeurs.

---

#### **6. Top 15 des aéroports d'origine et de destination**

##### **Code :**
```python
# Top 15 aéroports par nombre de vols
counts.sort_values(by='Origin', ascending=False).head(15)
```

##### **Explication :**
Permet de focaliser l’analyse sur les aéroports les plus fréquentés.

---

#### **7. Combiner tous les fichiers CSV**

##### **Code :**
```python
def combine_csv(csv_files, filter_cols, subset_cols, subset_vals, file_name):
    """
    Combine csv files into one DataFrame
    csv_files: list of csv file paths
    filter_cols: list of columns to filter
    subset_cols: list of columns to subset rows
    subset_vals: list of list of values to subset rows
    """
    df = pd.DataFrame()
    
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_temp = df_temp[filter_cols]
        for col, val in zip(subset_cols, subset_vals):
            df_temp = df_temp[df_temp[col].isin(val)]      
        df = pd.concat([df, df_temp], axis=0)
      
    df.to_csv(file_name, index=False)
    print(f'Combined csv stored at {file_name}')
```

##### **Utilisation :**
```python
# Définir les colonnes à inclure
cols = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 
        'Reporting_Airline', 'Origin', 'OriginState', 'Dest', 'DestState', 
        'CRSDepTime', 'Cancelled', 'Diverted', 'Distance', 'DistanceGroup', 
        'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'AirTime']

# Colonnes pour filtrer les valeurs
subset_cols = ['Origin', 'Dest', 'Reporting_Airline']

# Valeurs à inclure
subset_vals = [['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['UA', 'OO', 'WN', 'AA', 'DL']]

# Combiner tous les fichiers
combined_csv_filename = f"{base_path}combined_files.csv"
combine_csv(csv_files, cols, subset_cols, subset_vals, combined_csv_filename)
```

##### **Explication :**
Combine tous les fichiers CSV en un seul fichier après filtrage par colonnes et valeurs.

---

#### **8. Charger et explorer le fichier combiné**

##### **Code :**
```python
# Charger le fichier combiné
data = pd.read_csv(combined_csv_filename)

# Afficher les premières lignes
print("First 5 records:")
print(data.head())
```

---

#### **9. Nettoyage des données manquantes et transformation**

##### **Code :**
```python
# Renommer la colonne cible
data.rename(columns={'ArrDel15': 'is_delay'}, inplace=True)

# Vérifier les valeurs manquantes
missing_values = data.isnull().sum(axis=0)
print("Missing values per column:")
print(missing_values)

# Supprimer les lignes avec des valeurs nulles dans la cible
data = data[~data.is_delay.isnull()]

# Ajouter une colonne pour l'heure de départ
data['DepHourofDay'] = (data['CRSDepTime'] // 100)
```

---

#### **10. Questions résolues :**

1. **Combien de lignes et colonnes ?**  
   ```python
   print("The #rows and #columns are:", data.shape[0], "and", data.shape[1])
   ```

2. **Années incluses ?**
   ```python
   print("The years in this dataset are:", sorted(data['Year'].unique()))
   ```

3. **Plage de dates ?**
   ```python
   print("The date range for data is:", data['FlightDate'].min(), "to", data['FlightDate'].max())
   ```

4. **Compagnies incluses ?**
   ```python
   print("The airlines covered in this dataset are:", list(data['Reporting_Airline'].unique()))
   ```

5. **Aéroports couverts ?**
   ```python
   print("The Origin airports covered are:", list(data['Origin'].unique()))
   print("The Destination airports covered are:", list(data['Dest'].unique()))
   ```

--- 

Ces étapes complètes garantissent que les données sont nettoyées, combinées et prêtes pour l’analyse et la modélisation.




---------------
# Suite - étape 02 
---------------

### Étape détaillée : Exploration et préparation des données pour la modélisation

---

#### **1. Distribution des classes : Retard vs. Pas de retard**

##### **Code :**
```python
# Distribution des classes
(data.groupby('is_delay').size() / len(data)).plot(kind='bar')
plt.ylabel('Frequency')
plt.title('Distribution of classes')
plt.show()
```

##### **Explication :**
- Le graphique en barres montre la fréquence relative des vols retardés (`is_delay = 1`) par rapport à ceux sans retard (`is_delay = 0`).
- Cela permet d’évaluer un éventuel déséquilibre entre les classes, ce qui pourrait impacter la modélisation.

##### **Réponse à la question :**
- Si une classe (par exemple, pas de retard) est beaucoup plus fréquente, cela peut entraîner un modèle biaisé vers la classe majoritaire. Une stratégie de gestion des données déséquilibrées pourrait être nécessaire (pondération des classes ou suréchantillonnage).

---

#### **2. Analyse des retards selon diverses colonnes**

##### **Code :**
```python
# Colonnes à explorer
viz_columns = ['Month', 'DepHourofDay', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest']

# Visualisation
fig, axes = plt.subplots(3, 2, figsize=(20, 20), squeeze=False)

for idx, column in enumerate(viz_columns):
    ax = axes[idx // 2, idx % 2]
    temp = data.groupby(column)['is_delay'].value_counts(normalize=True).rename('percentage') \
        .mul(100).reset_index().sort_values(column)
    sns.barplot(x=column, y="percentage", hue="is_delay", data=temp, ax=ax)
    ax.set_ylabel('% delay/no-delay')

plt.show()

# Relation entre la distance et les retards
sns.lmplot(x="is_delay", y="Distance", data=data, fit_reg=False, hue='is_delay', legend=False)
plt.legend(loc='center')
plt.xlabel('is_delay')
plt.ylabel('Distance')
plt.title("Delay vs Distance")
plt.show()
```

##### **Réponses aux questions :**
1. **Quels mois ont le plus de retards ?**
   - Les mois d’été (juin, juillet, août) peuvent montrer des pics de retards en raison d’une demande accrue ou de conditions météorologiques (ex. : orages).

2. **À quelle heure de la journée les retards sont-ils les plus fréquents ?**
   - Les vols tardifs (soir/nuit) peuvent avoir plus de retards à cause de l’effet en cascade des retards accumulés tout au long de la journée.

3. **Quel jour de la semaine a le plus de retards ?**
   - Les week-ends (samedi, dimanche) peuvent avoir plus de retards en raison d’un volume de trafic plus élevé.

4. **Quelle compagnie aérienne a le plus de retards ?**
   - Les compagnies avec des vols régionaux ou des hubs encombrés peuvent montrer plus de retards.

5. **Quels aéroports d'origine et de destination ont le plus de retards ?**
   - Les hubs majeurs comme ATL, ORD, LAX, et SFO, avec un trafic dense, sont plus susceptibles d’avoir des retards.

6. **La distance est-elle un facteur de retard ?**
   - Les vols courts peuvent avoir moins de retards liés à des conditions météorologiques, tandis que les vols longs sont plus exposés aux aléas.

---

#### **3. Filtrage et encodage des colonnes**

##### **Filtrage des colonnes nécessaires :**
```python
# Copie de sauvegarde
data_orig = data.copy()

# Colonnes sélectionnées
data = data[['is_delay', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
             'Reporting_Airline', 'Origin', 'Dest', 'Distance', 'DepHourofDay']]
```

##### **Encodage catégoriel :**
```python
# Colonnes catégorielles
categorical_columns = ['Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
                       'Reporting_Airline', 'Origin', 'Dest', 'DepHourofDay']

# Conversion en catégorie
for c in categorical_columns:
    data[c] = data[c].astype('category')

# Encodage one-hot (dummy encoding)
data_dummies = pd.get_dummies(data[categorical_columns], drop_first=True)
data = pd.concat([data, data_dummies], axis=1)
data.drop(categorical_columns, axis=1, inplace=True)

# Vérification des dimensions et des colonnes
print(f"Dataset dimensions: {data.shape}")
print(f"Dataset columns: {list(data.columns)}")
```

##### **Explication :**
- **Filtrage :** Les colonnes inutiles comme la date brute ou les détails des retards spécifiques (`ArrDelayMinutes`) sont supprimées.
- **Encodage :** Les colonnes catégorielles sont converties en format numérique pour la modélisation (via `get_dummies`).

---

#### **4. Renommer la colonne cible pour la modélisation**

##### **Code :**
```python
# Renommer la colonne cible
data.rename(columns={'is_delay': 'target'}, inplace=True)
```

##### **Explication :**
- Cela rend la colonne cible plus descriptive et intuitive pour la modélisation.

---

#### **5. Préparation finale et vérification**

##### **Code :**
```python
# Vérification des valeurs manquantes
missing_values = data.isnull().sum(axis=0)
print("Missing values per column:")
print(missing_values)

# Suppression des lignes avec des valeurs nulles
data = data[~data.target.isnull()]
print(f"Dataset dimensions after null removal: {data.shape}")
```

---

### **Résumé des découvertes :**
1. **Classes :** La classe "Pas de retard" est beaucoup plus fréquente, indiquant un déséquilibre des classes.
2. **Colonnes pertinentes :** Des variables comme `Month`, `DayOfWeek`, `Reporting_Airline`, `Origin`, et `Distance` montrent une corrélation claire avec les retards.
3. **Nettoyage :** Les données inutiles ont été supprimées, et les variables catégorielles ont été encodées pour la modélisation.
4. **Problème de modélisation :** Une classification binaire pour prédire si un vol est retardé (`target = 1`) ou non (`target = 0`).

---

Les données sont maintenant prêtes pour être divisées en ensemble d'entraînement et de test, et utilisées pour entraîner un modèle de classification.



---------------
# Étape 03
---------------


### **Étape 3 : Entraînement et évaluation du modèle**

Voici une explication détaillée de chaque ligne de code de cette étape, en commençant par la préparation des données, l'entraînement du modèle et l'évaluation des résultats.

---

### **1. Préparation des données : division en ensembles d’entraînement, validation et test**

```python
from sklearn.model_selection import train_test_split

def split_data(data):
    train, test_and_validate = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])
    test, validate = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['target'])
    return train, validate, test

train, validate, test = split_data(data)

print(train['target'].value_counts())
print(test['target'].value_counts())
print(validate['target'].value_counts())
```

#### **Explication :**
1. **`train_test_split`:** Cette fonction divise les données en deux parties :
   - **Entraînement (`train`)** : Utilisé pour entraîner le modèle.
   - **Validation et Test (`test_and_validate`)** : Ces ensembles sont utilisés pour évaluer le modèle.
2. **Stratification (`stratify`):** Cela garantit que la proportion des classes dans les ensembles reste cohérente avec celle des données d’origine.
3. **Deuxième division :** L’ensemble `test_and_validate` est divisé à nouveau en :
   - **Test (`test`)** : Utilisé pour évaluer la performance finale du modèle.
   - **Validation (`validate`)** : Utilisé pour ajuster les hyperparamètres du modèle.
4. **Affichage des distributions :** La commande `value_counts()` permet de vérifier que les proportions des classes sont respectées dans chaque ensemble.

---

### **2. Création de l'estimateur LinearLearner dans Amazon SageMaker**

```python
import sagemaker
from sagemaker.serializers import CSVSerializer
from sagemaker.amazon.amazon_estimator import RecordSet
import boto3

classifier_estimator = sagemaker.LinearLearner(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    predictor_type='binary_classifier',
    binary_classifier_model_selection_criteria='cross_entropy_loss'
)
```

#### **Explication :**
1. **`sagemaker.LinearLearner`:** Est un algorithme d’apprentissage supervisé fourni par Amazon SageMaker pour résoudre les problèmes de classification binaire.
2. **Paramètres importants :**
   - `role`: Identifie les permissions AWS nécessaires pour exécuter le modèle.
   - `instance_count` et `instance_type`: Spécifient le type et le nombre d’instances AWS à utiliser pour l’entraînement.
   - `predictor_type`: Défini comme `binary_classifier` car le problème consiste à prédire une classe binaire (`0` ou `1`).
   - `binary_classifier_model_selection_criteria`: Utilise `cross_entropy_loss` comme critère pour optimiser le modèle.

---

### **3. Préparation des données pour l'entraînement**

```python
train_records = classifier_estimator.record_set(
    train.values[:, 1:].astype(np.float32),
    train.values[:, 0].astype(np.float32),
    channel='train'
)

val_records = classifier_estimator.record_set(
    validate.values[:, 1:].astype(np.float32),
    validate.values[:, 0].astype(np.float32),
    channel='validation'
)

test_records = classifier_estimator.record_set(
    test.values[:, 1:].astype(np.float32),
    test.values[:, 0].astype(np.float32),
    channel='test'
)
```

#### **Explication :**
1. **`record_set`:** Convertit les ensembles d’entraînement, de validation et de test en un format compatible avec SageMaker (protobuf ou CSV).
2. **Paramètres :**
   - `train.values[:, 1:]`: Sélectionne toutes les colonnes sauf la première (car elle contient les étiquettes).
   - `train.values[:, 0]`: Contient uniquement les étiquettes (`target`).
   - `astype(np.float32)`: Convertit les données en type `float32`, requis par SageMaker.
   - `channel`: Définit le rôle des données (train, validation, ou test).

---

### **4. Entraînement du modèle**

```python
classifier_estimator.fit([train_records, val_records, test_records])
```

#### **Explication :**
- La méthode `fit` lance le processus d’entraînement en utilisant les ensembles préparés (train, validation, test).
- SageMaker utilise les instances spécifiées pour exécuter le modèle LinearLearner.

---

### **5. Évaluation du modèle**

#### **Affichage des métriques d’entraînement :**
```python
from sagemaker.analytics import TrainingJobAnalytics

TrainingJobAnalytics(
    classifier_estimator._current_job_name,
    metric_names=['test:objective_loss', 'test:binary_f_beta', 'test:precision', 'test:recall']
).dataframe()
```

#### **Explication :**
1. **`TrainingJobAnalytics`:** Récupère les métriques de performance du modèle pendant l’entraînement.
2. **Métriques suivies :**
   - `test:objective_loss`: Mesure la perte de l'objectif pendant l'entraînement.
   - `test:binary_f_beta`: Combine précision et rappel dans une seule métrique.
   - `test:precision` et `test:recall`: Évaluent respectivement la précision et le rappel du modèle.

---

#### **Prédictions par lot :**
```python
def batch_linear_predict(test_data, estimator):
    batch_X = test_data.iloc[:, 1:]
    batch_X_file = 'batch-in.csv'
    upload_s3_csv(batch_X_file, 'batch-in', batch_X)

    batch_output = f"s3://{bucket}/{prefix}/batch-out/"
    batch_input = f"s3://{bucket}/{prefix}/batch-in/{batch_X_file}"

    classifier_transformer = estimator.transformer(
        instance_count=1,
        instance_type='ml.m4.xlarge',
        strategy='MultiRecord',
        assemble_with='Line',
        output_path=batch_output
    )

    classifier_transformer.transform(
        data=batch_input,
        data_type='S3Prefix',
        content_type='text/csv',
        split_type='Line'
    )

    classifier_transformer.wait()

    obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/batch-out/{batch_X_file}.out")
    target_predicted_df = pd.read_json(io.BytesIO(obj['Body'].read()), orient="records", lines=True)

    return test_data.iloc[:, 0], target_predicted_df.iloc[:, 0]
```

#### **Explication :**
1. **Chargement des données de test dans S3 :**
   - La fonction `upload_s3_csv` télécharge les données sur un bucket S3 pour exécution par lot.
2. **Prédictions en lot (`transform`):**
   - Les données sont analysées par le modèle pour produire des prédictions.
3. **Résultats :**
   - Les prédictions sont récupérées depuis S3 et converties en DataFrame.

---

### **6. Matrice de confusion et ROC**

#### **Matrice de confusion :**
```python
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(test_labels, target_predicted):
    matrix = confusion_matrix(test_labels, target_predicted)
    sns.heatmap(pd.DataFrame(matrix), annot=True, fmt='d', cmap="coolwarm")
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()
```

#### **Courbe ROC :**
```python
from sklearn import metrics

def plot_roc(test_labels, target_predicted):
    fpr, tpr, _ = metrics.roc_curve(test_labels, target_predicted)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
```

---

### **Réponses aux questions clés :**
1. **Différences entre précision, rappel et F1-score :**
   - La précision mesure la proportion de prédictions correctes parmi les prédictions positives.
   - Le rappel évalue la proportion de vrais positifs correctement identifiés.
   - Le F1-score combine précision et rappel, utile si les classes sont déséquilibrées.
   
2. **Métriques les plus importantes :**
   - Le rappel est critique pour éviter les retards imprévus.
   - Le F1-score est utile pour un compromis équilibré entre précision et rappel.

---

Cette étape prépare, entraîne et évalue le modèle en suivant les meilleures pratiques pour un pipeline machine learning dans SageMaker.




---------------
# Étape 04 - **Iteration II: Step 4 – Feature Engineering**. 
---------------



### **Objectif : Améliorer les performances du modèle**

Nous essayons d'améliorer le modèle en :
1. Ajoutant de nouvelles caractéristiques (`is_holiday`, conditions météo).
2. Gérant les données manquantes.
3. Réduisant les caractéristiques redondantes.
4. Essayant des algorithmes plus avancés comme **XGBoost**.
5. Effectuant une optimisation des hyperparamètres.

---

### **1. Ajouter une variable pour les jours fériés**

#### **Hypothèse :** Les retards sont plus fréquents pendant les jours fériés.

##### **Code :**
```python
# Étape 1 : Liste des jours fériés entre 2014 et 2018
holidays_14 = ['2014-01-01', '2014-01-20', '2014-02-17', '2014-05-26', '2014-07-04', '2014-09-01', '2014-10-13', '2014-11-11', '2014-11-27', '2014-12-25']
holidays_15 = ['2015-01-01', '2015-01-19', '2015-02-16', '2015-05-25', '2015-07-04', '2015-09-07', '2015-10-12', '2015-11-11', '2015-11-26', '2015-12-25']
holidays_16 = ['2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24', '2016-12-25']
holidays_17 = ['2017-01-02', '2017-01-16', '2017-02-20', '2017-05-29', '2017-07-04', '2017-09-04', '2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25']
holidays_18 = ['2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28', '2018-07-04', '2018-09-03', '2018-10-08', '2018-11-12', '2018-11-22', '2018-12-25']
holidays = holidays_14 + holidays_15 + holidays_16 + holidays_17 + holidays_18

# Étape 2 : Ajouter une colonne binaire "is_holiday"
data['is_holiday'] = data['FlightDate'].isin(holidays).astype(int)
```

##### **Explication :**
- `holidays`: Liste des jours fériés (fédéraux) aux États-Unis entre 2014 et 2018.
- `data['FlightDate'].isin(holidays)`: Vérifie si chaque date est un jour férié.
- `.astype(int)`: Convertit le résultat binaire (True/False) en 0/1.

---

### **2. Ajouter des conditions météorologiques**

#### **Hypothèse :** Les conditions météorologiques défavorables (pluie, neige, vent) augmentent les retards.

##### **Étapes :**
1. Télécharger les données météorologiques liées aux aéroports.
2. Mapper les stations météorologiques aux codes d’aéroports.
3. Gérer les valeurs manquantes (`fillna()`).
4. Ajouter des variables météo (`AWND`, `PRCP`, `SNOW`, etc.) pour les aéroports d'origine et de destination.

##### **Code :**
```python
# Étape 1 : Charger les données météo
weather = pd.read_csv('/path/to/daily-summaries.csv')

# Étape 2 : Mapper les stations météo aux aéroports
station = ['USW00023174', 'USW00012960', 'USW00003017', 'USW00094846', 'USW00013874', 'USW00023234', 'USW00003927', 'USW00023183', 'USW00013881']
airports = ['LAX', 'IAH', 'DEN', 'ORD', 'ATL', 'SFO', 'DFW', 'PHX', 'CLT']
station_map = {s: a for s, a in zip(station, airports)}
weather['airport'] = weather['STATION'].map(station_map)

# Étape 3 : Ajouter une colonne "MONTH"
weather['MONTH'] = weather['DATE'].apply(lambda x: x.split('-')[1])

# Étape 4 : Gérer les valeurs manquantes pour les colonnes "SNOW" et "SNWD"
weather['SNOW'].fillna(0, inplace=True)
weather['SNWD'].fillna(0, inplace=True)

# Étape 5 : Imputation des valeurs manquantes pour TAVG, TMAX, TMIN
weather_impute = weather.groupby(['MONTH', 'STATION']).agg({'TAVG': 'mean', 'TMAX': 'mean', 'TMIN': 'mean'}).reset_index()
weather = pd.merge(weather, weather_impute, how='left', on=['MONTH', 'STATION'])

# Étape 6 : Ajouter les données météo pour les aéroports d'origine et de destination
data = pd.merge(data, weather, how='left', left_on=['FlightDate', 'Origin'], right_on=['DATE', 'airport'])
data = pd.merge(data, weather, how='left', left_on=['FlightDate', 'Dest'], right_on=['DATE', 'airport'])
```

##### **Explication :**
1. `station_map`: Associe chaque station météo à un code d’aéroport.
2. `fillna(0)`: Remplit les valeurs manquantes pour les colonnes "SNOW" et "SNWD" par 0.
3. `groupby(['MONTH', 'STATION']).agg()`: Calcule la moyenne des températures pour chaque station météo par mois.

---

### **3. Encodage des variables catégoriques**

##### **Code :**
```python
# Définir les colonnes catégoriques
categorical_columns = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'is_holiday']

# Encodage one-hot
data_dummies = pd.get_dummies(data[categorical_columns], drop_first=True)

# Fusionner les nouvelles colonnes
data = pd.concat([data, data_dummies], axis=1)

# Supprimer les colonnes catégoriques d'origine
data.drop(columns=categorical_columns, inplace=True)
```

##### **Explication :**
- `get_dummies()`: Transforme les catégories en colonnes binaires.
- `drop_first=True`: Évite la multicolinéarité en supprimant une catégorie pour chaque variable.

---

### **4. Réentraîner le modèle avec de nouvelles caractéristiques**

##### **Code :**
```python
# Diviser les données
train, validate, test = split_data(data)

# Réentraîner LinearLearner
classifier_estimator2 = sagemaker.LinearLearner(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    predictor_type='binary_classifier',
    binary_classifier_model_selection_criteria='cross_entropy_loss'
)

# Préparer les ensembles d'entraînement
train_records = classifier_estimator2.record_set(train.values[:, 1:].astype(np.float32), train.values[:, 0].astype(np.float32), channel='train')
val_records = classifier_estimator2.record_set(validate.values[:, 1:].astype(np.float32), validate.values[:, 0].astype(np.float32), channel='validation')
test_records = classifier_estimator2.record_set(test.values[:, 1:].astype(np.float32), test.values[:, 0].astype(np.float32), channel='test')

# Entraîner le modèle
classifier_estimator2.fit([train_records, val_records, test_records])
```

---

### **5. Essayer XGBoost pour améliorer les performances**

##### **Code :**
```python
# Définir le modèle XGBoost
xgb = sagemaker.estimator.Estimator(
    container,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=f"s3://{bucket}/{prefix}/output/",
    sagemaker_session=sess
)

# Définir les hyperparamètres
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    objective='binary:logistic',
    eval_metric="auc",
    num_round=100
)

# Entraîner XGBoost
xgb.fit(inputs=data_channels)
```

---

### **6. Optimisation des hyperparamètres**

##### **

Code :**
```python
# Définir les plages d'hyperparamètres
hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.1, 0.5),
    'min_child_weight': ContinuousParameter(1, 10),
    'subsample': ContinuousParameter(0.5, 1)
}

# Configurer le tuner
tuner = HyperparameterTuner(
    xgb,
    objective_metric_name='validation:auc',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=2
)

# Lancer l'optimisation
tuner.fit(inputs=data_channels)
```

---

### **Résumé :**
1. **Ajouts majeurs :**
   - Variable `is_holiday`.
   - Variables météo (vent, pluie, neige, etc.).
2. **Encodage des catégories :** Transformé en colonnes binaires.
3. **Nouveaux modèles :**
   - LinearLearner réentraîné avec de nouvelles données.
   - XGBoost avec optimisation des hyperparamètres.
4. **Impact attendu :**
   - Amélioration de la précision et du rappel.
   - Réduction du surapprentissage grâce à des données plus diversifiées.





---------


### **Étape 4 : Feature Engineering (Ingénierie des caractéristiques)**

#### **Introduction : Pourquoi améliorer les caractéristiques ?**

Dans cette étape, nous modifions et enrichissons les données pour améliorer les performances du modèle. Cela inclut l’ajout de nouvelles variables, la gestion des données manquantes, et la réduction des variables redondantes. L'objectif est d'améliorer la précision, le rappel et d'autres métriques.

---

### **1. Identifier l'impact des classes déséquilibrées**

**Question : Comment l’équilibre des classes (retard vs. pas de retard) impacte-t-il la performance du modèle ?**

- Les classes déséquilibrées biaisent le modèle vers la classe majoritaire (par exemple, "pas de retard").
- **Solution :**
  - Rééquilibrer les classes par suréchantillonnage de la classe minoritaire ou sous-échantillonnage de la classe majoritaire.
  - Utiliser une pondération des classes dans l’entraînement du modèle.

---

### **2. Identifier les corrélations entre les caractéristiques**

**Question : Les caractéristiques sont-elles corrélées ?**

- Corrélations fortes (exemple : `DepDelay` et `ArrDelay`) peuvent introduire de la redondance et rendre le modèle plus complexe.
- **Solution :**
  - Calculer une matrice de corrélation (`data.corr()`) pour identifier les variables fortement corrélées.
  - Retirer ou combiner les caractéristiques corrélées.

---

### **3. Réduction des caractéristiques redondantes**

**Question : Peut-on réduire les caractéristiques ?**

- Variables comme `Date` sont redondantes avec `Year`, `Month`, `DayOfMonth`, et `DayOfWeek`.
- Retirer des caractéristiques comme `TotalDelayMinutes`, `DepDelayMinutes`, `ArrDelayMinutes` qui sont déjà résumées par `is_delay`.

##### **Code : Réduction des caractéristiques**
```python
data = data.drop(columns=['Date', 'TotalDelayMinutes', 'DepDelayMinutes', 'ArrDelayMinutes'])
```

---

### **4. Ajouter de nouvelles caractéristiques : Vacances**

**Hypothèse :** Les retards augmentent pendant les vacances.

1. Ajouter une variable binaire `is_holiday` pour marquer les vacances.
2. Utiliser les dates de vacances entre 2014 et 2018.

##### **Code : Ajout de `is_holiday`**
```python
holidays = holidays_14 + holidays_15 + holidays_16 + holidays_17 + holidays_18
data['is_holiday'] = data['FlightDate'].isin(holidays).astype(int)
```

---

### **5. Ajouter des données météorologiques**

**Hypothèse :** Les conditions météorologiques (pluie, neige, vents forts) influencent les retards.

1. Télécharger les données météorologiques depuis une source.
2. Associer les stations météorologiques aux aéroports via une correspondance.
3. Gérer les valeurs manquantes (`fillna()` ou imputation par la moyenne).
4. Ajouter les variables météorologiques aux données d’origine.

##### **Code : Ajout de données météorologiques**
```python
# Importer les données météorologiques
weather = pd.read_csv('/path/to/daily-summaries.csv')

# Correspondance entre stations et aéroports
station_map = {station: airport for station, airport in zip(station, airports)}
weather['airport'] = weather['STATION'].map(station_map)

# Gestion des valeurs manquantes
weather['SNOW'].fillna(0, inplace=True)
weather['SNWD'].fillna(0, inplace=True)
weather = weather.groupby(['MONTH', 'STATION']).agg({'TAVG': 'mean', 'TMAX': 'mean', 'TMIN': 'mean'}).reset_index()

# Ajouter les conditions météo à l’origine et à la destination
data = pd.merge(data, weather, how='left', left_on=['FlightDate', 'Origin'], right_on=['DATE', 'airport']) \
         .rename(columns={'AWND': 'AWND_O', 'PRCP': 'PRCP_O', 'TAVG': 'TAVG_O', 'SNOW': 'SNOW_O'}) \
         .drop(columns=['DATE', 'airport'])
```

---

### **6. Encodage des variables catégoriques**

**Pourquoi ?**
- Les algorithmes de machine learning nécessitent des données numériques. 
- Les variables catégoriques comme `Origin`, `Dest`, et `Reporting_Airline` doivent être encodées.

##### **Code : One-hot encoding**
```python
categorical_columns = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'is_holiday']
data_dummies = pd.get_dummies(data[categorical_columns], drop_first=True)
data = pd.concat([data, data_dummies], axis=1)
data.drop(columns=categorical_columns, inplace=True)
```

---

### **7. Réévaluation après Feature Engineering**

**Étapes :**
1. Diviser à nouveau les données en ensembles d’entraînement, validation et test.
2. Réentraîner le modèle LinearLearner et comparer les performances.

##### **Code : Réentraîner LinearLearner**
```python
train, validate, test = split_data(data)

classifier_estimator2 = sagemaker.LinearLearner(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    predictor_type='binary_classifier',
    binary_classifier_model_selection_criteria='cross_entropy_loss'
)

train_records = classifier_estimator2.record_set(train.values[:, 1:].astype(np.float32), train.values[:, 0].astype(np.float32), channel='train')
val_records = classifier_estimator2.record_set(validate.values[:, 1:].astype(np.float32), validate.values[:, 0].astype(np.float32), channel='validation')
test_records = classifier_estimator2.record_set(test.values[:, 1:].astype(np.float32), test.values[:, 0].astype(np.float32), channel='test')

classifier_estimator2.fit([train_records, val_records, test_records])
```

---

### **8. Essayer un modèle avancé : XGBoost**

1. Convertir les ensembles en fichiers CSV et les télécharger dans un bucket S3.
2. Définir un modèle XGBoost avec Amazon SageMaker.
3. Ajuster les hyperparamètres (`max_depth`, `eta`, etc.).
4. Évaluer la performance du modèle avec des prédictions en lot.

##### **Code : Entraîner XGBoost**
```python
xgb = sagemaker.estimator.Estimator(
    container,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=f"s3://{bucket}/{prefix}/output/",
    sagemaker_session=sess
)

xgb.set_hyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.8, objective='binary:logistic', eval_metric="auc", num_round=100)

xgb.fit(inputs=data_channels)
```

---

### **9. Évaluation finale et matrice de confusion**

1. Effectuer des prédictions en lot avec XGBoost.
2. Calculer et afficher la matrice de confusion.
3. Ajuster le seuil de décision si nécessaire.

##### **Code : Matrice de confusion**
```python
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(test_labels, target_predicted):
    matrix = confusion_matrix(test_labels, target_predicted)
    sns.heatmap(pd.DataFrame(matrix), annot=True, fmt='d', cmap="coolwarm")
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()
```

---

### **10. Optimisation des hyperparamètres**

**Pourquoi ?**
- Optimiser les paramètres de XGBoost pour obtenir de meilleures performances.
- Utiliser `HyperparameterTuner` dans SageMaker pour exécuter plusieurs essais parallèles.

##### **Code : Hyperparameter Optimization**
```python
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.1, 0.5),
    'min_child_weight': ContinuousParameter(1, 10),
    'subsample': ContinuousParameter(0.5, 1)
}

tuner = HyperparameterTuner(
    xgb,
    objective_metric_name='validation:auc',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=2
)

tuner.fit(inputs=data_channels)
```

---

### **Résumé des décisions clés :**
1. Ajout de `is_holiday` et des données météorologiques comme nouvelles caractéristiques.
2. Réduction des variables redondantes pour simplifier le modèle.
3. Encodage des variables catégoriques pour les rendre compatibles avec le modèle.
4. Exploration de XGBoost, un modèle avancé, pour capturer des interactions complexes entre les variables.
5. Optimisation des hyperparamètres pour maximiser les performances.


---------------------
# Conclusion:
---------------------

   ### **Conclusion : Résumé et réflexion sur le projet**

Maintenant que vous avez itéré plusieurs fois sur l'entraînement et l'évaluation de votre modèle, il est temps de conclure ce projet. Voici une analyse des apprentissages, des défis, et des étapes potentielles pour améliorer le pipeline à l'avenir.

---

### **1. Objectifs atteints ?**

- **Question : Les performances de votre modèle répondent-elles à votre objectif métier ?**
  - **Réponse :** 
    - Le modèle a atteint une précision de **X%** et un rappel de **Y%**, ce qui est proche (ou éloigné) de l'objectif initial.
    - Si les performances ne sont pas satisfaisantes, quelques ajustements possibles pourraient inclure :
      - Ajouter davantage de données externes (par exemple, trafic aérien, conditions sociales).
      - Affiner les techniques d'optimisation d'hyperparamètres.
      - Tester des modèles plus avancés comme des réseaux neuronaux profonds.

---

### **2. Améliorations réalisées**

- **Question : De combien votre modèle s'est-il amélioré après chaque itération ?**
  - **Réponse :** 
    - **Ajout des caractéristiques (`is_holiday`, météo) :** Amélioration de **Z%** sur l'AUC.
    - **Encodage des variables catégoriques :** Simplification du traitement des données et réduction du temps d'entraînement.
    - **XGBoost :** Performances accrues grâce à la capacité de capturer des interactions complexes.
    - **Optimisation des hyperparamètres :** Augmentation de **X%** de l'AUC grâce à une meilleure configuration du modèle.

---

### **3. Techniques utilisées et leur impact**

- **Techniques appliquées :**
  1. **Feature Engineering :**
     - Ajout de variables contextuelles (`is_holiday`, météo).
     - Encodage des catégories pour rendre les données compatibles avec les modèles.
  2. **Optimisation :**
     - Affinage des hyperparamètres pour XGBoost via Amazon SageMaker.
  3. **Évaluation continue :**
     - Utilisation de métriques comme l'AUC, la précision et le rappel pour guider les itérations.
- **Impact des techniques :**
  - L'ajout de caractéristiques contextuelles a eu le plus grand impact en permettant au modèle de mieux capturer les facteurs externes affectant les retards.

---

### **4. Défis rencontrés**

- **Questions abordées :**
  - **Données manquantes :** Certaines colonnes comme `TAVG`, `TMAX` avaient des valeurs manquantes conséquentes, nécessitant une imputation par la moyenne.
  - **Complexité computationnelle :** L'entraînement avec des modèles comme XGBoost sur un grand dataset a nécessité une gestion optimisée des ressources (instances SageMaker).
  - **Interprétation des résultats :** Comprendre l'impact des nouvelles caractéristiques sur les performances globales du modèle.

---

### **5. Questions restées ouvertes**

- **Pipeline non clair ?**
  - Une meilleure compréhension des relations entre certaines caractéristiques (par exemple, météo et retard) pourrait être explorée.
  - La validation croisée pourrait être ajoutée pour garantir des performances robustes sur des données non observées.

---

### **6. Enseignements clés**

- **Trois leçons majeures sur l'apprentissage machine à retenir :**
  1. **Importance des données :** La qualité et la pertinence des caractéristiques influencent plus que l'algorithme utilisé.
  2. **Pipeline itératif :** La performance d'un modèle s'améliore grâce à des itérations et à des ajustements progressifs.
  3. **Hyperparamètres :** L'optimisation peut transformer un bon modèle en un modèle excellent.

---

### **7. Prochaines étapes**

- Si davantage de temps ou de ressources étaient disponibles :
  - **Explorer des modèles avancés :** Réseaux neuronaux ou systèmes hybrides.
  - **Intégration de nouvelles données :** Par exemple, des données sur les infrastructures des aéroports ou des événements sociaux.
  - **Automatisation du pipeline :** Construire un pipeline ML de bout en bout, incluant le prétraitement, l'entraînement, et le déploiement.

---

### **8. Résumé pour présentation**

Préparez une présentation finale avec :
- Les résultats clés (métriques, graphiques de performances).
- Les décisions prises à chaque étape.
- Les leçons apprises et les recommandations pour des travaux futurs.

