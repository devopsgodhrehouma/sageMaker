# Récap 2

Voici une table récapitulative des principales métriques de performance en classification binaire, avec des explications basées sur l'exemple de patients malades ou sains :

```
+---------------------------+------------------------------------------------------------------------------------------+
|         Métrique          |                                  Explication                                             |
+---------------------------+------------------------------------------------------------------------------------------+
| Sensibilité               | Aussi appelée Recall ou Taux de Vrais Positifs (TVP).                                     |
|                           | Mesure la proportion de patients malades correctement identifiés                          |
|                           | par le modèle comme étant malades.                                                       |
|                           | *Exemple : Si 90 patients malades sur 100 sont détectés, la sensibilité est de 90%.*     |
+---------------------------+------------------------------------------------------------------------------------------+
| Spécificité               | Taux de Vrais Négatifs (TVN).                                                            |
|                           | Mesure la proportion de patients sains correctement identifiés                            |
|                           | par le modèle comme étant sains.                                                         |
|                           | *Exemple : Si 80 patients sains sur 100 sont correctement identifiés,                     |
|                           | la spécificité est de 80%.*                                                              |
+---------------------------+------------------------------------------------------------------------------------------+
| Précision                 | Valeur Prédictive Positive (VPP).                                                        |
|                           | Parmi les patients prédits malades, proportion qui sont réellement malades.              |
|                           | *Exemple : Si sur 50 patients prédits malades, 45 le sont réellement,                    |
|                           | la précision est de 90%.*                                                                |
+---------------------------+------------------------------------------------------------------------------------------+
| Valeur Prédictive         | (VPN). Parmi les patients prédits sains, proportion qui sont réellement sains.           |
| Négative                  |                                                                                          |
|                           | *Exemple : Si sur 70 patients prédits sains, 65 sont réellement sains,                   |
|                           | la VPN est d'environ 92.86%.*                                                            |
+---------------------------+------------------------------------------------------------------------------------------+
| Taux de Faux Positifs     | FPR (False Positive Rate).                                                              |
|                           | Proportion de patients sains incorrectement classés comme malades.                       |
|                           | *Exemple : Si 20 patients sains sur 100 sont classés comme malades,                      |
|                           | le taux de faux positifs est de 20%.*                                                    |
+---------------------------+------------------------------------------------------------------------------------------+
| Taux de Faux Négatifs     | FNR (False Negative Rate).                                                              |
|                           | Proportion de patients malades incorrectement classés comme sains.                       |
|                           | *Exemple : Si 10 patients malades sur 100 sont manqués, le taux de                       |
|                           | faux négatifs est de 10%.*                                                               |
+---------------------------+------------------------------------------------------------------------------------------+
| Taux de Faux Découvertes  | FDR (False Discovery Rate).                                                             |
|                           | Parmi les patients prédits malades, proportion qui sont en réalité sains.                |
|                           | *Exemple : Si sur 50 prédits malades, 5 sont sains, le FDR est de 10%.*                  |
+---------------------------+------------------------------------------------------------------------------------------+
| Précision Globale         | Exactitude du modèle. Pourcentage total de patients correctement classés                 |
|                           | (malades et sains).                                                                      |
|                           | *Exemple : Si sur 200 patients, 170 sont correctement classés,                            |
|                           | la précision globale est de 85%.*                                                        |
+---------------------------+------------------------------------------------------------------------------------------+
```

**Légende des termes :**

- **Patients malades** : Patients qui ont réellement la maladie (cas positifs).
- **Patients sains** : Patients qui n'ont pas la maladie (cas négatifs).
- **Correctement identifiés** : Prédictions du modèle qui correspondent à la réalité.
- **Incorrectement classés** : Erreurs du modèle, où la prédiction ne correspond pas à la réalité.

Cette table vous aide à comprendre chaque métrique en lien avec l'exemple des patients, facilitant ainsi la compréhension de la performance du modèle dans un contexte médical.
