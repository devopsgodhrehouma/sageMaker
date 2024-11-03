### Lab 3.4 - Entraînement d’un Modèle de Machine Learning

### Aperçu du Lab

Dans ce lab, vous allez continuer d’explorer le dataset biomécanique de la colonne vertébrale. Vous allez d’abord diviser le dataset en trois ensembles de données distincts pour l’entraînement, la validation et le test. Ensuite, vous utiliserez ces données pour entraîner un modèle de machine learning (ML) avec l’algorithme XGBoost.

### Objectifs

À la fin de ce lab, vous serez capable de :

- Diviser des données en ensembles d’entraînement, de validation et de test
- Entraîner un modèle XGBoost dans Amazon SageMaker

### Prérequis

Ce lab nécessite :

- Un ordinateur portable avec accès Wi-Fi, sous Microsoft Windows, macOS, ou Linux (Ubuntu, SUSE ou Red Hat)
- Pour les utilisateurs de Microsoft Windows : l'accès administrateur sur l'ordinateur
- Un navigateur internet tel que Chrome, Firefox ou IE9 (les versions précédentes d'Internet Explorer ne sont pas supportées)

### Durée

Ce lab nécessite environ 30 minutes. L'environnement restera actif pendant 120 minutes.

### Accéder à la Console de Gestion AWS

1. En haut de ces instructions, choisissez **Start Lab** pour lancer votre lab.
2. Un panneau **Start Lab** s’ouvre et affiche le statut du lab.
3. Attendez que le message **Lab status: ready** apparaisse, puis fermez le panneau en cliquant sur **X**.
4. En haut de ces instructions, choisissez **AWS**.
5. Cela ouvrira la Console de gestion AWS dans un nouvel onglet de navigateur. Vous serez automatiquement connecté.

*Astuce : Si un nouvel onglet ne s'ouvre pas, un bandeau ou une icône peut indiquer que votre navigateur empêche les fenêtres contextuelles. Sélectionnez ce bandeau ou cette icône, puis choisissez d'autoriser les pop-ups.*

Organisez l'onglet de la Console de gestion AWS de manière à pouvoir voir les deux fenêtres côte à côte pour faciliter le suivi des étapes du lab.

---

### Tâche 1 : Accéder à une instance de notebook dans Amazon SageMaker

Dans cette tâche, vous allez ouvrir l’environnement JupyterLab et accéder au notebook pour compléter le lab.

#### Pour ouvrir JupyterLab :

1. Dans la Console de gestion AWS, dans le menu **Services**, sélectionnez **Amazon SageMaker**.
2. Dans le menu de navigation de gauche, développez la section **Notebook** et sélectionnez **Instances de notebook**.
3. Recherchez l’instance de notebook nommée **MyNotebook**. Ouvrez l'instance de notebook JupyterLab en cliquant sur **Open JupyterLab** à la fin de la ligne correspondante.

---

### Tâche 2 : Ouvrir un notebook dans votre instance de notebook

Dans cette tâche, vous allez ouvrir le notebook pour ce lab.

1. Dans l’environnement JupyterLab, allez dans le navigateur de fichiers dans le panneau de gauche et localisez le fichier **3_4-machinelearning.ipynb**.
2. Ouvrez le fichier **en_us/3_4-machinelearning.ipynb** en le sélectionnant.

*Astuce : Si une fenêtre apparaît vous demandant de sélectionner un noyau, choisissez **conda_python3**, puis cliquez sur **Select**.*

3. Suivez les instructions dans le notebook.

---

### Conclusion

Vous avez maintenant réussi à :

- Diviser les données en ensembles d’entraînement, de validation et de test
- Entraîner un modèle XGBoost dans Amazon SageMaker

### Lab terminé

Félicitations ! Vous avez terminé le lab.

1. Pour confirmer que vous souhaitez terminer le lab, en haut de cette page, choisissez **End Lab**, puis confirmez en sélectionnant **Yes**.
2. Un panneau devrait apparaître avec le message : **DELETE has been initiated...** Vous pouvez fermer cette boîte de message.


