# Table des Hyperparamètres et Analogies - hyperparamètres pour XGboost

- https://neptune.ai/blog/xgboost-vs-lightgbm

| **Hyperparamètre**            | **Définition en Machine Learning**                                                                                   | **Analogie de la Vie Courante**                                                           |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Taux d'Apprentissage**      | Vitesse à laquelle le modèle ajuste ses paramètres en réponse à l'erreur observée.                                    | Vitesse à laquelle on apprend une nouvelle compétence (par ex., apprendre à danser).      |
| *(Learning Rate)*             |                                                                                                                      |                                                                                           |
| **Nombre d'Itérations**       | Nombre de fois que le modèle parcourt l'ensemble des données pour apprendre (nombre d'époques).                       | Nombre de répétitions pour maîtriser une recette ou une compétence (pratiquer un sport).  |
| *(Epochs ou Num_Round)*       |                                                                                                                      |                                                                                           |
| **Profondeur de l'Arbre**     | Complexité maximale des arbres de décision dans le modèle, affectant sa capacité à capturer des relations complexes.  | Niveau de détail dans un projet (décorer une pièce de manière simple ou élaborée).        |
| *(Max Depth)*                 |                                                                                                                      |                                                                                           |
| **Régularisation**            | Technique pour éviter que le modèle ne s'adapte trop étroitement aux données d'entraînement (éviter le surapprentissage). | Éviter de surcharger une recette (ne pas ajouter trop d'épices pour garder l'équilibre).  |
| *(Alpha, Lambda)*             |                                                                                                                      |                                                                                           |
| **Sous-échantillonnage**      | Proportion des données utilisées pour entraîner le modèle, aidant à réduire la variance et le surapprentissage.       | Goûter de petites portions pendant la cuisson pour ajuster le goût sans tout manger.      |
| *(Subsample)*                 |                                                                                                                      |                                                                                           |
| **Taux de Décroissance**      | Réduction progressive du taux d'apprentissage au cours du temps pour affiner l'apprentissage.                          | Commencer à apprendre vite puis ralentir pour se concentrer sur les détails (apprendre une langue). |
| *(Learning Rate Decay)*       |                                                                                                                      |                                                                                           |
| **Taille des Lots**           | Nombre d'exemples utilisés pour mettre à jour les paramètres du modèle à chaque itération.                            | Préparer plusieurs plats à la fois ou un seul à la fois (efficacité vs. attention aux détails). |
| *(Batch Size)*                |                                                                                                                      |                                                                                           |

---

**Légende :**

- **Hyperparamètre** : Réglage défini avant l'entraînement du modèle qui influence le processus d'apprentissage.
- **Définition en Machine Learning** : Rôle et impact de l'hyperparamètre sur le modèle.
- **Analogie de la Vie Courante** : Exemple concret pour illustrer le concept de manière accessible.

---

**Explications Supplémentaires :**

- **Taux d'Apprentissage (Learning Rate)**
  - *Machine Learning* : Un taux élevé peut entraîner une convergence rapide mais risque de manquer le minimum global. Un taux faible assure une convergence plus stable mais peut être lent.
  - *Analogie* : Si tu apprends trop vite, tu risques de faire des erreurs. Si tu apprends trop lentement, tu peux t'ennuyer ou perdre du temps.

- **Nombre d'Itérations (Epochs ou Num_Round)**
  - *Machine Learning* : Trop d'itérations peuvent conduire à un surapprentissage, trop peu à un sous-apprentissage.
  - *Analogie* : Répéter une recette trop de fois peut devenir lassant, pas assez de fois et tu ne la maîtrises pas.

- **Profondeur de l'Arbre (Max Depth)**
  - *Machine Learning* : Une plus grande profondeur permet au modèle de capturer des relations complexes mais augmente le risque de surapprentissage.
  - *Analogie* : Une décoration trop complexe peut surcharger la pièce, tandis qu'une décoration trop simple peut manquer de caractère.

- **Régularisation (Alpha, Lambda)**
  - *Machine Learning* : Pénalise les coefficients élevés pour éviter que le modèle ne s'adapte trop aux données d'entraînement.
  - *Analogie* : Ajouter trop d'épices peut dominer le plat. Une quantité équilibrée donne un meilleur résultat.

- **Sous-échantillonnage (Subsample)**
  - *Machine Learning* : Utiliser une partie des données pour chaque mise à jour réduit la variance et aide à généraliser.
  - *Analogie* : En goûtant de petites portions, tu peux ajuster la recette sans te rassasier avant le repas.

- **Taux de Décroissance du Taux d'Apprentissage (Learning Rate Decay)**
  - *Machine Learning* : Réduire le taux d'apprentissage au fil du temps permet un ajustement plus fin à la fin de l'entraînement.
  - *Analogie* : Au début, tu peux apprendre rapidement les bases d'une langue, puis ralentir pour maîtriser la grammaire complexe.

- **Taille des Lots (Batch Size)**
  - *Machine Learning* : Une grande taille de lot rend l'apprentissage plus stable mais nécessite plus de mémoire. Une petite taille de lot introduit plus de bruit mais peut généraliser mieux.
  - *Analogie* : Préparer plusieurs plats en même temps peut être efficace mais risqué si tu manques d'attention.

---

Cette table vise à rendre les concepts de machine learning accessibles en les associant à des situations familières. Ainsi, même sans connaissances techniques, on peut comprendre l'importance et l'impact des hyperparamètres sur le processus d'apprentissage d'un modèle.



# Partie -02 (hyperparamètres pour CNN et Autoencodeurs)

Voici une liste des **hyperparamètres** couramment utilisés dans les réseaux de neurones, en particulier dans les **réseaux de neurones convolutifs (CNN)** et les **autoencodeurs**, avec des explications pour chacun :

---

### **Table des Hyperparamètres pour les Réseaux de Neurones (CNN et Autoencodeurs)**

| **Hyperparamètre**            | **Définition**                                                                                 | **Impact sur le Modèle**                                                                    |
|-------------------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **Taux d'Apprentissage**      | Vitesse à laquelle le modèle met à jour ses poids en réponse à l'erreur calculée.              | Un taux élevé accélère l'apprentissage mais peut rendre le modèle instable. Un taux faible assure une convergence stable mais peut ralentir l'apprentissage. |
| **Taille des Lots**           | Nombre d'exemples d'entraînement utilisés pour une mise à jour des poids du modèle.            | Les petits lots introduisent plus de bruit mais peuvent aider à généraliser. Les grands lots rendent l'apprentissage plus stable mais nécessitent plus de mémoire. |
| **Nombre d'Époques**          | Nombre de fois que le modèle parcourt l'ensemble des données d'entraînement.                   | Plus d'époques permettent au modèle d'apprendre davantage, mais risquent le surapprentissage. Trop peu peuvent conduire à un sous-apprentissage. |
| **Fonction d'Activation**     | Fonction utilisée pour introduire la non-linéarité dans le modèle (ReLU, Sigmoïde, Tanh, etc.). | Affecte la capacité du modèle à capturer des relations complexes. Certaines activations conviennent mieux à certaines tâches. |
| **Nombre de Couches**         | Profondeur du réseau, c'est-à-dire le nombre total de couches (convolutives, denses, etc.).    | Un réseau plus profond peut capturer des caractéristiques plus complexes mais est plus difficile à entraîner. |
| **Nombre de Filtres**         | Pour les couches convolutives, le nombre de filtres ou de kernels appliqués.                   | Plus de filtres peuvent extraire plus de caractéristiques, mais augmentent le coût computationnel. |
| **Taille du Filtre**          | Dimensions du kernel de convolution (par exemple, 3x3, 5x5).                                    | Affecte la taille des caractéristiques détectées. Les petits filtres capturent des détails fins, les grands filtres capturent des caractéristiques globales. |
| **Stride (Pas)**              | Nombre de pixels par lequel le filtre est déplacé lors de la convolution.                      | Un stride plus grand réduit la taille des sorties et peut diminuer le coût computationnel, mais peut perdre des informations. |
| **Padding (Remplissage)**     | Ajout de pixels autour de l'image pour contrôler la taille de la sortie après convolution.     | Le padding permet de préserver la taille des entrées, évitant la réduction des dimensions après convolution. |
| **Taux de Dropout**           | Fraction des unités à ignorer pendant l'entraînement pour éviter le surapprentissage.           | Un taux plus élevé peut réduire le surapprentissage, mais trop élevé peut conduire à un sous-apprentissage. |
| **Fonction de Perte**         | Fonction utilisée pour mesurer l'erreur du modèle (par exemple, MSE, Entropie Croisée).         | Le choix de la fonction de perte influence la manière dont le modèle apprend et optimise ses poids. |
| **Optimiseur**                | Algorithme utilisé pour mettre à jour les poids du modèle (SGD, Adam, RMSprop, etc.).          | Différents optimiseurs peuvent converger plus rapidement ou gérer mieux certains types de problèmes. |
| **Coefficient de Régularisation L1/L2** | Paramètres qui pénalisent les poids élevés pour éviter le surapprentissage.            | Aide à garder les poids du modèle petits, favorisant la simplicité et la généralisation. |
| **Dimensions Latentes**       | Pour les autoencodeurs, la taille de la couche cachée centrale qui représente les données compressées. | Une dimension latente plus petite force le modèle à apprendre une représentation plus compacte, mais peut perdre des informations importantes. |
| **Taux de Décroissance du Taux d'Apprentissage** | Réduction progressive du taux d'apprentissage au cours du temps. | Aide le modèle à affiner son apprentissage en réduisant la taille des mises à jour des poids au fil des époques. |

---

### **Explications Supplémentaires :**

- **Taux d'Apprentissage (Learning Rate)** :

  - **Ce que c'est** : C'est la vitesse à laquelle le modèle ajuste ses poids en réponse à l'erreur mesurée.
  - **Impact** : Un taux trop élevé peut faire diverger l'apprentissage (le modèle "saute" les minima de la fonction de perte). Un taux trop faible ralentit l'apprentissage et peut coincer le modèle dans des minima locaux.
  - **Analogie** : C'est comme apprendre une nouvelle compétence. Si tu apprends trop vite, tu risques de mal comprendre. Trop lentement, et tu perds du temps.

- **Taille des Lots (Batch Size)** :

  - **Ce que c'est** : Nombre d'échantillons passés dans le réseau avant la mise à jour des poids.
  - **Impact** : Les petits lots rendent l'apprentissage plus bruité mais peuvent aider à généraliser. Les grands lots sont plus stables mais nécessitent plus de mémoire.
  - **Analogie** : Étudier seul (petit lot) vs en groupe (grand lot). Étudier seul peut être plus flexible mais moins stable.

- **Nombre d'Époques (Epochs)** :

  - **Ce que c'est** : Nombre de fois que l'algorithme voit l'ensemble des données.
  - **Impact** : Trop d'époques peuvent entraîner un surapprentissage, trop peu un sous-apprentissage.
  - **Analogie** : Répéter un exercice trop de fois peut mener à l'épuisement, pas assez de fois et tu ne le maîtrises pas.

- **Fonction d'Activation** :

  - **Ce que c'est** : Introduit la non-linéarité dans le réseau, permettant de modéliser des relations complexes.
  - **Impact** : Le choix affecte la capacité du réseau à apprendre. Par exemple, ReLU est souvent utilisé dans les CNN.
  - **Analogie** : Choisir le bon outil pour le bon travail, comme utiliser un tournevis plutôt qu'un marteau pour visser.

- **Nombre de Couches et de Filtres** :

  - **Ce que c'est** : La profondeur du réseau et le nombre de filtres déterminent sa capacité à extraire des caractéristiques.
  - **Impact** : Plus de couches/filtres augmentent la capacité du modèle mais aussi le risque de surapprentissage.
  - **Analogie** : Construire une maison avec plus d'étages offre plus d'espace mais nécessite plus de ressources.

- **Taille du Filtre et Stride** :

  - **Ce que c'est** : Déterminent comment le modèle parcourt les données d'entrée pour extraire des caractéristiques.
  - **Impact** : Affectent la résolution des caractéristiques capturées et la taille des sorties.
  - **Analogie** : Utiliser une loupe (petit filtre) pour voir les détails fins ou une vue d'ensemble (grand filtre) pour voir la globalité.

- **Dropout** :

  - **Ce que c'est** : Technique pour éviter le surapprentissage en désactivant aléatoirement des neurones pendant l'entraînement.
  - **Impact** : Force le réseau à être plus robuste et à ne pas dépendre d'unités spécifiques.
  - **Analogie** : Comme pratiquer un sport sous différentes conditions pour être prêt à tout.

- **Optimiseur** :

  - **Ce que c'est** : Algorithme qui détermine comment les poids sont mis à jour (ex. SGD, Adam).
  - **Impact** : Affecte la vitesse de convergence et la capacité à échapper aux minima locaux.
  - **Analogie** : Choisir le meilleur itinéraire pour atteindre une destination, en évitant les embouteillages.

- **Dimensions Latentes (Autoencodeurs)** :

  - **Ce que c'est** : Taille de la représentation compressée des données.
  - **Impact** : Une dimension trop petite peut perdre des informations, trop grande n'apporte pas de compression.
  - **Analogie** : Résumer un livre en quelques phrases. Trop court, et tu perds des détails importants.

- **Taux de Décroissance du Taux d'Apprentissage** :

  - **Ce que c'est** : Réduit le taux d'apprentissage au fil du temps pour stabiliser l'entraînement.
  - **Impact** : Permet d'affiner l'apprentissage une fois les grandes lignes apprises.
  - **Analogie** : Comme ralentir sa voiture en approchant de sa destination pour mieux manœuvrer.

---

### **Pourquoi ces Hyperparamètres sont-ils Importants ?**

- **Ajustement Fin** : Ils permettent d'adapter le modèle aux spécificités des données et du problème à résoudre.
- **Éviter le Sur/Sous-apprentissage** : Un bon réglage aide à trouver un équilibre entre un modèle trop simple (sous-apprentissage) et trop complexe (surapprentissage).
- **Performance et Efficacité** : Influencent le temps d'entraînement, l'utilisation de la mémoire et la précision du modèle.

---

### **Comment Choisir et Ajuster les Hyperparamètres ?**

1. **Comprendre le Problème** : Différents problèmes nécessitent différents réglages. Par exemple, la reconnaissance d'images complexes peut bénéficier de réseaux plus profonds.

2. **Expérimentation** : Tester différentes valeurs pour voir comment elles affectent les performances.

3. **Validation Croisée** : Utiliser des ensembles de validation pour évaluer les performances et éviter le surapprentissage.

4. **Techniques d'Optimisation** :

   - **Recherche en Grille** : Tester systématiquement des combinaisons prédéfinies.
   - **Recherche Aléatoire** : Essayer des combinaisons au hasard.
   - **Optimisation Bayésienne** : Utiliser des méthodes statistiques pour choisir les hyperparamètres à tester.

---

### **Conclusion**

Les hyperparamètres sont essentiels pour construire des réseaux de neurones performants. En comprenant leur rôle et leur impact, vous pouvez ajuster votre modèle pour obtenir les meilleurs résultats possibles. Cela nécessite souvent un équilibre entre l'expérimentation pratique et une compréhension théorique du fonctionnement des réseaux de neurones.

---

N'hésitez pas à expérimenter avec ces hyperparamètres pour voir comment ils affectent les performances de votre modèle. Chaque problème est unique, et ce qui fonctionne pour l'un peut ne pas fonctionner pour un autre.
