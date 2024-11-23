# ANNEXE : Explication détaillée du code et des fonctions

Dans cette annexe, nous allons passer en revue chaque fonction du code fourni, en détaillant leur utilité, leur fonctionnement interne, les paramètres qu'elles prennent et les résultats qu'elles produisent. L'objectif est de comprendre en profondeur chaque étape du processus de préparation des données, de modélisation et d'évaluation pour la prédiction des retards d'avion.

---

## 1. Fonction `zip2csv`

### Description

Cette fonction a pour but d'extraire les fichiers CSV contenus dans des fichiers ZIP. Étant donné que les données initiales sont compressées pour économiser de l'espace, cette fonction facilite l'accès aux données en les décompressant.

### Code

```python
def zip2csv(zipFile_name, file_path):
    """
    Extrait les fichiers CSV des fichiers ZIP.

    Paramètres:
    - zipFile_name: le nom du fichier ZIP à extraire.
    - file_path: le chemin du dossier où les fichiers CSV seront stockés.

    Retour:
    - Aucun. Les fichiers CSV sont extraits et stockés dans le dossier spécifié.
    """
    try:
        with ZipFile(zipFile_name, 'r') as z:
            print(f'Extraction de {zipFile_name}')
            z.extractall(path=file_path)
    except:
        print(f'Échec de l'extraction pour {zipFile_name}')
```

### Explication détaillée

- **Importation nécessaire**: La fonction utilise le module `ZipFile` de la bibliothèque `zipfile` pour manipuler les fichiers ZIP.
  
- **Paramètres**:
  - `zipFile_name`: le chemin complet du fichier ZIP à extraire.
  - `file_path`: le chemin du dossier où les fichiers extraits seront stockés.

- **Processus**:
  1. **Ouverture du fichier ZIP**: La fonction utilise un gestionnaire de contexte `with` pour ouvrir le fichier ZIP en lecture (`'r'`).
  2. **Extraction des fichiers**: La méthode `extractall()` est utilisée pour extraire tous les fichiers du ZIP vers le chemin spécifié par `file_path`.
  3. **Gestion des exceptions**: Si une erreur survient lors de l'extraction, une exception est capturée, et un message d'erreur est affiché.

- **Sortie**: Les fichiers sont extraits dans le dossier spécifié, et un message de confirmation est imprimé. En cas d'erreur, un message d'échec est affiché.

### Exemple d'utilisation

```python
zip_files = ['data1.zip', 'data2.zip']
csv_base_path = '/path/to/csv_files/'

for file in zip_files:
    zip2csv(file, csv_base_path)
```

---

## 2. Fonction `combine_csv`

### Description

La fonction `combine_csv` est conçue pour combiner plusieurs fichiers CSV en un seul DataFrame pandas, en appliquant des filtres sur les colonnes et les lignes. Elle permet de réduire la taille des données en ne conservant que les informations pertinentes pour l'analyse.

### Code

```python
def combine_csv(csv_files, filter_cols, subset_cols, subset_vals, file_name):
    """
    Combine plusieurs fichiers CSV en un seul DataFrame après filtrage.

    Paramètres:
    - csv_files: liste des chemins des fichiers CSV à combiner.
    - filter_cols: liste des colonnes à conserver.
    - subset_cols: liste des colonnes sur lesquelles appliquer des filtres pour sous-ensemble.
    - subset_vals: liste de listes des valeurs à conserver pour chaque colonne de `subset_cols`.
    - file_name: nom du fichier CSV de sortie combiné.

    Retour:
    - Aucun. Le DataFrame combiné est enregistré sous forme de fichier CSV.
    """
    df = pd.DataFrame()

    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_temp = df_temp[filter_cols]
        for col, val in zip(subset_cols, subset_vals):
            df_temp = df_temp[df_temp[col].isin(val)]
        df = pd.concat([df, df_temp], axis=0)

    df.to_csv(file_name, index=False)
    print(f'Fichier CSV combiné enregistré sous {file_name}')
```

### Explication détaillée

- **Importation nécessaire**: La fonction utilise la bibliothèque `pandas` pour manipuler les DataFrames.

- **Paramètres**:
  - `csv_files`: liste des fichiers CSV à traiter.
  - `filter_cols`: colonnes à conserver dans les DataFrames.
  - `subset_cols`: colonnes sur lesquelles appliquer des filtres pour sélectionner des sous-ensembles de données.
  - `subset_vals`: valeurs à conserver pour chaque colonne de `subset_cols`. Il s'agit d'une liste de listes, chaque sous-liste correspondant aux valeurs pour une colonne spécifique.
  - `file_name`: nom du fichier de sortie où le DataFrame combiné sera enregistré.

- **Processus**:
  1. **Initialisation**: Création d'un DataFrame vide `df` pour stocker les données combinées.
  2. **Boucle sur les fichiers CSV**:
     - **Lecture du fichier**: Chaque fichier CSV est lu dans `df_temp`.
     - **Filtrage des colonnes**: On conserve uniquement les colonnes spécifiées dans `filter_cols`.
     - **Filtrage des lignes**: Pour chaque paire `col`, `val` dans `subset_cols` et `subset_vals`:
       - On utilise `df_temp[col].isin(val)` pour conserver les lignes où la colonne `col` contient des valeurs dans `val`.
     - **Concaténation**: Le DataFrame filtré `df_temp` est concaténé avec le DataFrame principal `df`.
  3. **Enregistrement du fichier**: Le DataFrame combiné `df` est enregistré sous forme de fichier CSV sans index.

- **Sortie**: Le fichier CSV combiné est enregistré avec le nom spécifié, et un message de confirmation est affiché.

### Exemple d'utilisation

```python
cols = ['Year', 'Month', 'DayofMonth', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest', 'ArrDel15']
subset_cols = ['Origin', 'Dest', 'Reporting_Airline']
subset_vals = [['ATL', 'ORD'], ['LAX', 'SFO'], ['UA', 'DL']]

combine_csv(csv_files, cols, subset_cols, subset_vals, 'combined_data.csv')
```

---

## 3. Fonction `split_data`

### Description

Cette fonction divise le DataFrame en ensembles d'entraînement, de validation et de test. Elle utilise une division stratifiée pour assurer une répartition équilibrée des classes cibles dans chaque ensemble.

### Code

```python
from sklearn.model_selection import train_test_split

def split_data(data):
    """
    Divise les données en ensembles d'entraînement, de validation et de test.

    Paramètres:
    - data: le DataFrame pandas contenant les données à diviser.

    Retour:
    - train: DataFrame d'entraînement.
    - validate: DataFrame de validation.
    - test: DataFrame de test.
    """
    train, test_and_validate = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['target']
    )
    test, validate = train_test_split(
        test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['target']
    )
    return train, validate, test
```

### Explication détaillée

- **Importation nécessaire**: La fonction utilise `train_test_split` du module `sklearn.model_selection`.

- **Paramètres**:
  - `data`: DataFrame à diviser.

- **Processus**:
  1. **Première division**:
     - On divise `data` en `train` (80%) et `test_and_validate` (20%).
     - `stratify=data['target']` assure que la proportion des classes cibles est maintenue dans chaque ensemble.
     - `random_state=42` pour la reproductibilité des résultats.
  2. **Seconde division**:
     - On divise `test_and_validate` en `test` (10%) et `validate` (10%).
     - La stratification est à nouveau appliquée sur `test_and_validate['target']`.
  3. **Retour**:
     - La fonction retourne les trois ensembles : `train`, `validate`, `test`.

- **Sortie**: Trois DataFrames contenant respectivement les données d'entraînement, de validation et de test.

### Exemple d'utilisation

```python
train, validate, test = split_data(data)
```

---

## 4. Fonction `batch_linear_predict`

### Description

Cette fonction effectue des prédictions en lot sur un ensemble de données de test en utilisant un modèle entraîné avec Amazon SageMaker. Elle prépare les données, lance une tâche de transformation par lot et récupère les résultats.

### Code

```python
def batch_linear_predict(test_data, estimator):
    """
    Effectue des prédictions en lot sur les données de test.

    Paramètres:
    - test_data: DataFrame des données de test.
    - estimator: l'estimateur SageMaker entraîné.

    Retour:
    - test_labels: les étiquettes réelles des données de test.
    - target_predicted: les prédictions du modèle.
    """
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

    s3 = boto3.client('s3')
    obj = s3.get_object(
        Bucket=bucket, Key=f"{prefix}/batch-out/{batch_X_file}.out"
    )
    target_predicted_df = pd.read_json(
        io.BytesIO(obj['Body'].read()), orient="records", lines=True
    )
    return test_data.iloc[:, 0], target_predicted_df.iloc[:, 0]
```

### Explication détaillée

- **Importations nécessaires**:
  - `boto3`: pour interagir avec AWS S3.
  - `io`: pour la manipulation des flux de données.
  - `pandas`: pour manipuler les DataFrames.

- **Paramètres**:
  - `test_data`: le DataFrame contenant les données de test.
  - `estimator`: l'estimateur (modèle) entraîné avec SageMaker.

- **Processus**:
  1. **Préparation des données**:
     - `batch_X` contient les caractéristiques (features) en excluant la colonne cible.
     - Le DataFrame est enregistré dans un fichier CSV `batch-in.csv`.
     - Le fichier est téléchargé sur S3 à l'aide de la fonction `upload_s3_csv`.
  2. **Configuration des chemins S3**:
     - `batch_output`: dossier de sortie sur S3 où les prédictions seront stockées.
     - `batch_input`: chemin S3 du fichier d'entrée pour la prédiction.
  3. **Création du transformateur**:
     - `classifier_transformer` est un objet qui gère les prédictions en lot.
     - Les paramètres spécifient le type d'instance, la stratégie d'assemblage et le chemin de sortie.
  4. **Lancement de la transformation**:
     - La méthode `transform()` lance le travail de prédiction en lot.
     - Les paramètres précisent le type de données et la façon dont elles sont divisées.
     - La méthode `wait()` attend la fin du travail.
  5. **Récupération des prédictions**:
     - On utilise `boto3` pour accéder au fichier de sortie sur S3.
     - Le fichier est lu et converti en DataFrame `target_predicted_df`.
     - La fonction retourne les étiquettes réelles et les prédictions.

- **Sortie**: Les étiquettes réelles `test_labels` et les prédictions `target_predicted`.

### Exemple d'utilisation

```python
test_labels, target_predicted = batch_linear_predict(test, classifier_estimator)
```

---

## 5. Fonction `plot_confusion_matrix`

### Description

Cette fonction génère une matrice de confusion pour visualiser les performances du modèle en termes de prédictions correctes et incorrectes.

### Code

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(test_labels, target_predicted):
    """
    Affiche la matrice de confusion des prédictions.

    Paramètres:
    - test_labels: les étiquettes réelles.
    - target_predicted: les étiquettes prédites par le modèle.

    Retour:
    - Aucun. La matrice de confusion est affichée.
    """
    matrix = confusion_matrix(test_labels, target_predicted)
    df_confusion = pd.DataFrame(matrix)
    colormap = sns.color_palette("BrBG", 10)
    sns.heatmap(df_confusion, annot=True, fmt='d', cbar=None, cmap=colormap)
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    plt.ylabel("Classe Réelle")
    plt.xlabel("Classe Prédite")
    plt.show()
```

### Explication détaillée

- **Importations nécessaires**:
  - `confusion_matrix` de `sklearn.metrics`.
  - `seaborn` et `matplotlib.pyplot` pour la visualisation.

- **Paramètres**:
  - `test_labels`: les étiquettes réelles du jeu de données de test.
  - `target_predicted`: les étiquettes prédites par le modèle.

- **Processus**:
  1. **Calcul de la matrice de confusion**:
     - La fonction `confusion_matrix` calcule la matrice de confusion entre les étiquettes réelles et prédites.
  2. **Création du DataFrame**:
     - La matrice est convertie en DataFrame pour faciliter la visualisation.
  3. **Personnalisation du colormap**:
     - Un palette de couleurs est définie pour améliorer l'esthétique du graphique.
  4. **Affichage de la heatmap**:
     - La fonction `sns.heatmap` est utilisée pour afficher la matrice de confusion.
     - Les annotations sont activées pour afficher les valeurs numériques.
     - Les axes sont étiquetés pour indiquer les classes réelles et prédites.
  5. **Affichage du graphique**:
     - La méthode `plt.show()` affiche la figure générée.

- **Sortie**: La matrice de confusion est affichée à l'écran.

### Exemple d'utilisation

```python
plot_confusion_matrix(test_labels, target_predicted)
```

---

## 6. Fonction `plot_roc`

### Description

Cette fonction calcule et affiche la courbe ROC (Receiver Operating Characteristic) pour évaluer les performances du modèle en termes de taux de vrais positifs et de faux positifs. Elle calcule également plusieurs métriques de performance.

### Code

```python
from sklearn import metrics

def plot_roc(test_labels, target_predicted):
    """
    Calcule et affiche la courbe ROC et les métriques de performance.

    Paramètres:
    - test_labels: les étiquettes réelles.
    - target_predicted: les étiquettes prédites par le modèle.

    Retour:
    - Aucun. La courbe ROC et les métriques sont affichées.
    """
    TN, FP, FN, TP = confusion_matrix(test_labels, target_predicted).ravel()
    # Calcul des métriques
    Sensitivity = float(TP)/(TP+FN)*100
    Specificity = float(TN)/(TN+FP)*100
    Precision = float(TP)/(TP+FP)*100
    NPV = float(TN)/(TN+FN)*100
    FPR = float(FP)/(FP+TN)*100
    FNR = float(FN)/(TP+FN)*100
    FDR = float(FP)/(TP+FP)*100
    ACC = float(TP+TN)/(TP+FP+FN+TN)*100

    # Affichage des métriques
    print("Sensibilité (TPR): ", Sensitivity, "%")
    print("Spécificité (TNR): ", Specificity, "%")
    print("Précision: ", Precision, "%")
    print("Valeur Prédictive Négative: ", NPV, "%")
    print("Taux de Faux Positifs (FPR): ", FPR, "%")
    print("Taux de Faux Négatifs (FNR): ", FNR, "%")
    print("Taux de Fausses Découvertes (FDR): ", FDR, "%")
    print("Exactitude (Accuracy): ", ACC, "%")

    # Calcul de l'AUC
    print("Validation AUC", metrics.roc_auc_score(test_labels, target_predicted))

    # Calcul des points pour la courbe ROC
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, target_predicted)
    roc_auc = metrics.auc(fpr, tpr)

    # Tracé de la courbe ROC
    plt.figure()
    plt.plot(fpr, tpr, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Ligne diagonale pour la référence
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Caractéristique de Fonctionnement du Récepteur (ROC)')
    plt.legend(loc="lower right")

    # Ajout des seuils sur un second axe
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, linestyle='dashed', color='red')
    ax2.set_ylabel('Seuil', color='red')
    ax2.set_ylim([thresholds[-1], thresholds[0]])
    ax2.set_xlim([fpr[0], fpr[-1]])

    plt.show()
```

### Explication détaillée

- **Importations nécessaires**:
  - `metrics` de `sklearn` pour les calculs de performance.
  - `matplotlib.pyplot` pour la visualisation.

- **Paramètres**:
  - `test_labels`: étiquettes réelles.
  - `target_predicted`: étiquettes prédites.

- **Processus**:
  1. **Extraction des valeurs de la matrice de confusion**:
     - La fonction `confusion_matrix` est utilisée, et `ravel()` permet de récupérer TN, FP, FN, TP.
  2. **Calcul des métriques de performance**:
     - **Sensibilité (TPR)**: proportion de vrais positifs correctement identifiés.
     - **Spécificité (TNR)**: proportion de vrais négatifs correctement identifiés.
     - **Précision**: proportion de prédictions positives correctes.
     - **Valeur Prédictive Négative (NPV)**: proportion de prédictions négatives correctes.
     - **Taux de Faux Positifs (FPR)**: proportion de faux positifs parmi tous les négatifs réels.
     - **Taux de Faux Négatifs (FNR)**: proportion de faux négatifs parmi tous les positifs réels.
     - **Taux de Fausses Découvertes (FDR)**: proportion de faux positifs parmi les prédictions positives.
     - **Exactitude (Accuracy)**: proportion totale de prédictions correctes.
  3. **Affichage des métriques**:
     - Les métriques calculées sont affichées avec des pourcentages.
  4. **Calcul de l'AUC**:
     - L'aire sous la courbe ROC est calculée avec `roc_auc_score`.
  5. **Calcul des points ROC**:
     - `roc_curve` retourne les taux de faux positifs, les taux de vrais positifs et les seuils utilisés.
  6. **Tracé de la courbe ROC**:
     - La courbe ROC est tracée en utilisant `fpr` et `tpr`.
     - Une ligne diagonale est tracée pour représenter un classificateur aléatoire.
     - Les axes sont étiquetés, et le titre est défini.
  7. **Ajout des seuils**:
     - Un second axe Y est créé pour afficher les seuils correspondants.
     - Les seuils sont tracés en fonction du `fpr`.
  8. **Affichage du graphique**:
     - La figure est affichée avec `plt.show()`.

- **Sortie**: Les métriques de performance sont affichées, et la courbe ROC est tracée.

### Exemple d'utilisation

```python
plot_roc(test_labels, target_predicted)
```

---

## 7. Fonctions pour le Modèle XGBoost

### Description

Dans le code, plusieurs fonctions et étapes sont dédiées à la préparation des données et à l'entraînement du modèle XGBoost avec Amazon SageMaker.

### Préparation des données pour XGBoost

#### Fonction `upload_s3_csv`

```python
def upload_s3_csv(filename, folder, dataframe):
    """
    Télécharge un DataFrame sous forme de fichier CSV vers un dossier S3 spécifié.

    Paramètres:
    - filename: nom du fichier CSV.
    - folder: dossier S3 où le fichier sera stocké.
    - dataframe: le DataFrame pandas à télécharger.

    Retour:
    - Aucun. Le fichier est téléchargé sur S3.
    """
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, header=False, index=False)
    s3_resource.Bucket(bucket).Object(
        os.path.join(prefix, folder, filename)
    ).put(Body=csv_buffer.getvalue())
```

### Explication détaillée

- **Importations nécessaires**:
  - `io.StringIO` pour manipuler des flux en mémoire.
  - `boto3` pour interagir avec S3.

- **Paramètres**:
  - `filename`: nom du fichier CSV à créer.
  - `folder`: chemin du dossier dans le bucket S3.
  - `dataframe`: le DataFrame à convertir en CSV.

- **Processus**:
  1. **Conversion en CSV**:
     - `dataframe.to_csv()` convertit le DataFrame en CSV, stocké dans `csv_buffer`.
  2. **Téléchargement sur S3**:
     - Le contenu de `csv_buffer` est envoyé au chemin S3 spécifié.
     - La méthode `put()` de l'objet S3 est utilisée pour télécharger le fichier.

- **Sortie**: Le DataFrame est enregistré en tant que fichier CSV sur S3.

### Exemple d'utilisation

```python
upload_s3_csv('train.csv', 'train', train)
```

---

### Création des Input Channels pour SageMaker

```python
train_channel = sagemaker.inputs.TrainingInput(
    f"s3://{bucket}/{prefix}/train/",
    content_type='text/csv'
)

validate_channel = sagemaker.inputs.TrainingInput(
    f"s3://{bucket}/{prefix}/validate/",
    content_type='text/csv'
)

data_channels = {'train': train_channel, 'validation': validate_channel}
```

### Explication détaillée

- **Utilité**:
  - Ces objets spécifient les emplacements des données d'entraînement et de validation pour SageMaker.
  - Ils sont utilisés lors de l'appel à `fit()` pour entraîner le modèle.

- **Paramètres**:
  - `sagemaker.inputs.TrainingInput` prend le chemin S3 et le type de contenu.

- **Processus**:
  - Les chemins S3 sont formatés pour pointer vers les dossiers contenant les données.
  - `data_channels` est un dictionnaire regroupant les canaux de données.

### Entraînement du modèle XGBoost

```python
from sagemaker.image_uris import retrieve
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')

xgb = sagemaker.estimator.Estimator(
    container,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m4.xlarge',
    output_path=f's3://{bucket}/{prefix}/output/',
    sagemaker_session=sess
)

xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    silent=0,
    objective='binary:logistic',
    eval_metric='auc',
    num_round=100
)

xgb.fit(inputs=data_channels)
```

### Explication détaillée

- **Récupération du conteneur**:
  - La fonction `retrieve` obtient l'URI du conteneur Docker pour XGBoost.

- **Configuration de l'estimateur**:
  - `Estimator` est configuré avec le conteneur, le rôle IAM, le type d'instance et le chemin de sortie.
  - `set_hyperparameters` définit les hyperparamètres du modèle XGBoost.

- **Entraînement du modèle**:
  - La méthode `fit` est appelée avec les canaux de données spécifiés.

---

## Conclusion

Nous avons passé en revue les principales fonctions du code, en détaillant leur utilité et leur fonctionnement interne. Chaque fonction joue un rôle crucial dans le pipeline de préparation des données, d'entraînement du modèle et d'évaluation des performances. En comprenant ces fonctions, vous serez mieux équipé pour adapter et améliorer le modèle en fonction de vos besoins spécifiques.

---

**Remarque**: N'oubliez pas de gérer les ressources AWS utilisées (instances, tâches de formation, stockage S3) pour éviter des coûts imprévus. Assurez-vous de bien comprendre chaque étape du processus et de tester le code sur des échantillons de données avant de l'appliquer à l'ensemble complet.
