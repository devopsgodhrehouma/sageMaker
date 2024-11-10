# Explication 

Dans ce notebook, nous allons apprendre à déployer un modèle de machine learning entraîné, à effectuer des prédictions, puis à exécuter une transformation par lot pour prédire sur un ensemble de données complet. Nous travaillerons avec Amazon SageMaker, un service qui facilite le développement, l'entraînement et le déploiement de modèles de machine learning à grande échelle.

## Aperçu du scénario

Vous travaillez pour un prestataire de soins de santé et votre objectif est d'améliorer la détection des anomalies chez les patients orthopédiques. Pour cela, vous utiliserez le machine learning pour prédire si un patient présente une anomalie en fonction de certaines caractéristiques biomécaniques.

## À propos du jeu de données

Le jeu de données que nous utilisons est le **"Vertebral Column"** de l'UCI Machine Learning Repository. Il contient des mesures biomécaniques de patients, avec une étiquette indiquant s'ils sont "normaux" ou "anormaux".

### Caractéristiques du jeu de données :

- **Pelvic incidence** (Incidence pelvienne)
- **Pelvic tilt** (Inclinaison pelvienne)
- **Lumbar lordosis angle** (Angle de lordose lombaire)
- **Sacral slope** (Pente sacrée)
- **Pelvic radius** (Rayon pelvien)
- **Grade of spondylolisthesis** (Degré de spondylolisthésis)
- **Classe** : 0 pour "Normal", 1 pour "Anormal"

## Configuration du lab

Avant de commencer, nous devons configurer notre environnement :

1. **Importation des bibliothèques nécessaires** : Nous importons des bibliothèques Python comme pandas pour la manipulation des données, boto3 pour interagir avec AWS, sagemaker pour utiliser SageMaker, etc.

2. **Définition du bucket S3** : Nous spécifions un bucket S3 pour stocker nos données. Ce bucket est un espace de stockage en ligne sur AWS.

3. **Téléchargement et préparation des données** :

   - Nous téléchargeons le fichier zip contenant le jeu de données depuis l'UCI Machine Learning Repository.
   - Nous extrayons le fichier ARFF (un format de fichier pour les datasets) et le chargeons dans un DataFrame pandas.
   - Nous remplaçons les étiquettes de classe par des nombres : 0 pour "Normal" et 1 pour "Anormal".
   - Nous réorganisons les colonnes pour placer la classe en première position.

4. **Division des données** :

   - Nous divisons le jeu de données en trois parties : entraînement (train), test et validation.
   - La division est faite de manière aléatoire mais stratifiée, ce qui signifie que la proportion de classes est maintenue dans chaque ensemble.

5. **Téléchargement des données sur S3** :

   - Nous écrivons chaque ensemble (train, test, validation) dans des fichiers CSV et les téléchargeons dans des dossiers spécifiques sur S3.

6. **Configuration du modèle XGBoost** :

   - Nous récupérons l'image Docker de XGBoost fournie par SageMaker.
   - Nous définissons les hyperparamètres du modèle, comme le nombre de tours (num_round), la métrique d'évaluation (eval_metric), et l'objectif (objective).

7. **Entraînement du modèle** :

   - Nous créons un objet Estimator de SageMaker pour XGBoost avec les paramètres définis.
   - Nous spécifions les canaux de données pour l'entraînement et la validation.
   - Nous lançons l'entraînement avec la méthode `fit()`.

## Étape 1 : Hébergement du modèle

Une fois le modèle entraîné, nous voulons le déployer pour qu'il puisse faire des prédictions sur de nouvelles données.

- **Déploiement du modèle** :

  - Nous utilisons la méthode `deploy()` sur l'objet du modèle entraîné (`xgb_model`).
  - Nous spécifions le nombre d'instances (ici, 1) et le type d'instance (par exemple, 'ml.m4.xlarge').
  - Le déploiement crée un endpoint (point de terminaison) sur lequel nous pouvons envoyer des requêtes pour obtenir des prédictions.

## Étape 2 : Effectuer des prédictions

Maintenant que le modèle est déployé, nous pouvons lui envoyer des données et obtenir des prédictions.

1. **Préparation des données de test** :

   - Nous examinons notre ensemble de test qui contient 31 instances.
   - Nous sélectionnons une ligne (par exemple, la première) et retirons la colonne 'class' car nous voulons prédire cette valeur.

2. **Envoi de la requête de prédiction** :

   - Nous convertissons la ligne en format CSV sous forme de chaîne de caractères.
   - Nous utilisons le prédicteur (`xgb_predictor`) pour envoyer la donnée et obtenir une prédiction.

3. **Interprétation du résultat** :

   - Le modèle renvoie une probabilité plutôt qu'une classe 0 ou 1.
   - Par exemple, une sortie de `0.9966071844100952` indique une haute probabilité que le patient soit 'anormal'.

4. **Vérification de la précision** :

   - Nous comparons la prédiction avec la valeur réelle dans notre ensemble de test pour vérifier si le modèle prédit correctement.

5. **Challenge facultatif** :

   - Nous pouvons essayer avec d'autres lignes de données pour voir si le modèle est cohérent dans ses prédictions.

## Étape 3 : Terminer le modèle déployé

Comme nous sommes facturés pour les ressources utilisées, il est important de supprimer le endpoint lorsque nous n'en avons plus besoin.

- **Suppression du endpoint** :

  - Nous utilisons la méthode `delete_endpoint()` sur le prédicteur pour supprimer le endpoint et libérer les ressources.

## Étape 4 : Effectuer une transformation par lot (Batch Transform)

Au lieu de faire des prédictions une par une, nous pouvons utiliser le Batch Transform de SageMaker pour prédire sur tout un ensemble de données en une seule opération.

1. **Préparation des données pour Batch Transform** :

   - Nous prenons l'ensemble de test complet et retirons la colonne 'class' pour ne garder que les caractéristiques.
   - Nous enregistrons ce DataFrame en tant que fichier CSV et le téléchargeons sur S3.

2. **Configuration du Batch Transform** :

   - Nous spécifions l'emplacement d'entrée (notre fichier CSV sur S3) et de sortie (un dossier sur S3 où les résultats seront stockés).
   - Nous créons un objet transformateur à partir de notre modèle (`xgb_model.transformer`).

3. **Exécution du Batch Transform** :

   - Nous appelons la méthode `transform()` sur le transformateur en lui passant les informations nécessaires.
   - SageMaker va automatiquement :

     - Lancer une instance avec le modèle déployé.
     - Appliquer le modèle sur toutes les données d'entrée.
     - Stocker les prédictions dans l'emplacement de sortie sur S3.
     - Terminer l'instance une fois le travail terminé.

4. **Téléchargement et analyse des résultats** :

   - Nous récupérons le fichier de sortie depuis S3.
   - Nous chargeons les prédictions dans un DataFrame pandas.
   - Les prédictions sont des probabilités.

5. **Conversion des probabilités en classes** :

   - Nous définissons une fonction pour convertir les probabilités en classes binaires (0 ou 1) en utilisant un seuil (par exemple, 0.65).
   - Nous appliquons cette fonction aux prédictions pour obtenir les classes prédites.

6. **Comparaison avec les valeurs réelles** :

   - Nous comparons les classes prédites avec les classes réelles dans l'ensemble de test.
   - Cela nous permet d'évaluer la performance du modèle.

## Points importants à retenir

- **Machine Learning supervisé** : Nous utilisons des données étiquetées (avec des classes connues) pour entraîner notre modèle.

- **Préparation des données** : La qualité des données et leur préparation (nettoyage, division, etc.) sont cruciales pour un bon modèle.

- **Entraînement du modèle** : Nous utilisons XGBoost, un algorithme puissant pour les problèmes de classification et de régression.

- **Déploiement** : Nous pouvons déployer notre modèle sur un endpoint pour des prédictions en temps réel ou utiliser Batch Transform pour des prédictions sur de grands ensembles de données.

- **Gestion des ressources** : Il est important de gérer les ressources cloud pour éviter des coûts inutiles (par exemple, en supprimant les endpoints inutilisés).

## Prochaines étapes

- **Évaluation du modèle** : Dans un prochain lab, nous calculerons des métriques pour évaluer la performance de notre modèle (précision, rappel, etc.).

- **Amélioration du modèle** : Nous pourrons ajuster les hyperparamètres ou essayer d'autres algorithmes pour améliorer les prédictions.

## Conclusion

Ce notebook vous a guidé à travers le processus complet de préparation des données, entraînement d'un modèle, déploiement pour des prédictions en temps réel, et exécution de prédictions par lot. En comprenant chaque étape, vous êtes mieux préparé pour développer et déployer vos propres modèles de machine learning en utilisant Amazon SageMaker.
