# Tutoriel Guidé : Amazon SageMaker et Importation de Données

Ce tutoriel vous guide dans la configuration d'un environnement Amazon SageMaker pour créer et utiliser des notebooks Jupyter, télécharger et extraire des données provenant d'une source externe, et charger ces données dans un DataFrame `pandas` pour les analyser.

---

### Aperçu du Lab

Dans ce lab, vous apprendrez à :
1. Lancer une instance de notebook Amazon SageMaker.
2. Créer et manipuler un notebook Jupyter.
3. Télécharger des données depuis une source externe, les extraire, et les charger dans un DataFrame.
4. Enregistrer votre travail localement pour pouvoir continuer avec la même configuration ultérieurement.

### Objectifs
À la fin de ce lab, vous serez capable de :
- Lancer une instance de notebook SageMaker.
- Créer et naviguer dans un notebook Jupyter.
- Exécuter du code dans le notebook pour télécharger et traiter des données.
- Enregistrer et télécharger votre notebook pour le réutiliser plus tard.

### Prérequis
- Un ordinateur avec Windows, macOS ou Linux et un navigateur comme Chrome, Firefox ou Edge.
- Accès à un compte AWS pour utiliser Amazon SageMaker.

### Durée
Environ 30 minutes. L’environnement restera actif pendant 180 minutes.

---

## Guide Étape par Étape

### Étape 1 : Lancer l'Instance de Notebook SageMaker

1. **Accéder à la Console AWS** :
   - Rendez-vous sur la Console de gestion AWS.
   - Cliquez sur **Démarrer le Lab** pour lancer votre lab. Attendez que le statut soit "prêt".
   - Une fois prêt, sélectionnez **AWS** pour ouvrir la Console de gestion AWS dans un nouvel onglet.

2. **Naviguer vers SageMaker** :
   - Dans la Console AWS, allez dans **Services** et sélectionnez **Amazon SageMaker**.

3. **Créer une Instance de Notebook** :
   - Dans SageMaker, étendez la section **Notebook** dans le menu de gauche et sélectionnez **Instances de notebook**.
   - Cliquez sur **Créer une instance de notebook**.
   - Donnez un nom à l’instance de notebook, par exemple `MonNotebook`.
   - Choisissez **ml.m4.xlarge** comme type d'instance.
   - Sous **Identifiant de plateforme**, sélectionnez **notebook-al2-v1** pour utiliser Amazon Linux 2 comme système d’exploitation.
   - Dans **Configuration du cycle de vie**, choisissez la configuration qui inclut **ml-pipeline**.
   - Laissez les autres paramètres par défaut et cliquez sur **Créer une instance de notebook**.
   - Le statut de l’instance de notebook sera d'abord "En attente". Attendez qu’il passe à "En service".

4. **Ouvrir JupyterLab** :
   - Une fois l’instance de notebook en service, cliquez sur **Ouvrir JupyterLab** pour lancer l'environnement du notebook.

### Étape 2 : Explorer JupyterLab

1. **Naviguer dans l'interface de JupyterLab** :
   - Familiarisez-vous avec l'environnement JupyterLab. Il comprend un menu principal, une barre latérale de navigation pour les fichiers, et une zone de travail pour ouvrir des notebooks et d’autres fichiers.
   - Dans le navigateur de fichiers, cherchez **PythonCheatSheet.ipynb** et ouvrez-le pour un récapitulatif rapide des commandes Python.

2. **Barre de Menu et Outils** :
   - **Fichier** : Enregistrer, rétablir ou créer des points de contrôle.
   - **Édition** : Modifier la structure du notebook (couper, copier, coller des cellules).
   - **Affichage** : Activer ou désactiver les options du notebook.
   - **Exécuter** : Exécuter les cellules dans le notebook.
   - **Noyau** : Changer l’environnement de programmation.
   - **Aide** : Accéder aux ressources d’aide.

3. **Exécuter des Cellules** :
   - Exécutez du code en sélectionnant une cellule et en appuyant sur **SHIFT + ENTRÉE**.
   - Les cellules peuvent être de différents types : code, Markdown (pour du texte), et raw (brut).

### Étape 3 : Ouvrir un Notebook d’Exemple

1. **Explorer les Exemples SageMaker** :
   - Dans JupyterLab, changez la vue de navigation de gauche pour **Amazon SageMaker Samples**.
   - Recherchez le notebook d’exemple **linear_learner_mnist.ipynb**.
   - Cliquez sur **Créer une copie** pour le dupliquer dans votre espace de travail et pouvoir le modifier.

### Étape 4 : Importer des Données

Les données utilisées dans ce lab proviennent d'un dataset sur la colonne vertébrale de l'UC Irvine Machine Learning Repository.

1. **Créer un Nouveau Notebook** :
   - Dans JupyterLab, allez dans **Fichier > Nouveau > Notebook**.
   - Choisissez **conda_python3** pour le noyau et cliquez sur **Sélectionner**.
   - Dans la première cellule, appuyez sur **M** sur votre clavier pour passer en Markdown, puis entrez :
     ```markdown
     # Importation des Données
     ```
   - Appuyez sur **SHIFT + ENTRÉE** pour afficher la cellule Markdown.

2. **Télécharger et Extraire les Données** :
   - Dans une nouvelle cellule de code, ajoutez les imports suivants et désactivez les avertissements :
     ```python
     import warnings, requests, zipfile, io
     warnings.simplefilter('ignore')
     import pandas as pd
     from scipy.io import arff
     ```
   - Pour télécharger les données, ajoutez une nouvelle cellule de code et entrez :
     ```python
     f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
     r = requests.get(f_zip, stream=True)
     Vertebral_zip = zipfile.ZipFile(io.BytesIO(r.content))
     Vertebral_zip.extractall()
     ```
   - Sélectionnez les deux cellules de code, puis appuyez sur **SHIFT + ENTRÉE** pour les exécuter. Vérifiez le panneau de navigation à gauche ; quatre nouveaux fichiers devraient apparaître :
     - **column_2C_weka.arff**
     - **column_2C.dat**
     - **column_3C_weka.arff**
     - **column_3C.dat**

3. **Examiner et Charger les Données** :
   - Cliquez sur chaque fichier pour voir son contenu.
   - Pour charger les données dans un DataFrame, créez une nouvelle cellule de code et entrez :
     ```python
     data = arff.loadarff('column_2C_weka.arff')
     df = pd.DataFrame(data[0])
     df.head()
     ```
   - Exécutez la cellule pour afficher les premières lignes du DataFrame.

### Étape 5 : Enregistrer Votre Notebook (Optionnel)

1. **Télécharger le Notebook** :
   - Dans le navigateur de fichiers, faites un clic droit sur le notebook que vous souhaitez enregistrer et choisissez **Télécharger**.
   - Sélectionnez un emplacement sur votre ordinateur pour l'enregistrer.

2. **Continuer le Travail Plus Tard** :
   - Si vous voulez continuer à travailler sur le notebook lors d'une prochaine session, relancez l’environnement SageMaker, téléchargez le notebook et reprenez.

---

### Conclusion

Félicitations ! Vous avez :
- Lancé une instance de notebook SageMaker.
- Créé et exploré un notebook Jupyter.
- Téléchargé, extrait et chargé des données externes pour les analyser.

Pour terminer le lab :
1. Sélectionnez **Terminer le Lab** en haut de la page et confirmez en sélectionnant **Oui**.
2. Un message vous indiquera la fin du lab.
