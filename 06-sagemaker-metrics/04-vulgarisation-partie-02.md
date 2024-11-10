# Récap 1
Je vous présente une table récapitulative des principales métriques de performance en classification binaire, 
avec une brève explication pour chacune :

```
+-------------------------+--------------------------------------------------------------------------------------------+
|       Métrique          |                                Explication                                                 |
+-------------------------+--------------------------------------------------------------------------------------------+
| Sensibilité             | Capacité du modèle à identifier correctement les cas positifs (détecter les anomalies).    |
| Spécificité             | Capacité du modèle à identifier correctement les cas négatifs (reconnaître les normaux).   |
| Précision               | Proportion des prédictions positives qui sont correctes (fiabilité des anomalies détectées).|
| Valeur Prédictive       | Proportion des prédictions négatives qui sont correctes (fiabilité des normaux détectés).  |
| Négative (VPN)          |                                                                                            |
| Taux de Faux Positifs   | Proportion des cas négatifs incorrectement classés comme positifs (fausses alertes).       |
| Taux de Faux Négatifs   | Proportion des cas positifs manqués par le modèle (anomalies non détectées).               |
| Taux de Faux Découvertes| Proportion des prédictions positives qui sont incorrectes (erreurs parmi les anomalies     |
|                         | détectées).                                                                                |
| Précision Globale       | Pourcentage total de prédictions correctes du modèle (exactitude générale).                |
+-------------------------+--------------------------------------------------------------------------------------------+
```

**Légende des termes :**

- **Sensibilité** : Aussi appelée **Recall** ou **Taux de Vrais Positifs** (TVP). Elle mesure la proportion de cas positifs correctement identifiés par le modèle.

- **Spécificité** : Aussi appelée **Taux de Vrais Négatifs** (TVN). Elle mesure la proportion de cas négatifs correctement identifiés.

- **Précision** : Aussi appelée **Valeur Prédictive Positive** (VPP). Elle indique la fiabilité des prédictions positives du modèle.

- **Valeur Prédictive Négative (VPN)** : Indique la fiabilité des prédictions négatives du modèle.

- **Taux de Faux Positifs** : Aussi appelé **Faux Positif Rate** (FPR). C'est le taux auquel le modèle génère de fausses alertes.

- **Taux de Faux Négatifs** : Aussi appelé **Faux Négatif Rate** (FNR). Il mesure la fréquence à laquelle le modèle manque des cas positifs.

- **Taux de Faux Découvertes** : Aussi appelé **False Discovery Rate** (FDR). Il indique la proportion de prédictions positives qui sont incorrectes.

- **Précision Globale** : Aussi appelée **Exactitude**. C'est le pourcentage total de prédictions correctes effectuées par le modèle.

Cette table vous aide à comprendre rapidement ce que représente chaque métrique dans l'évaluation d'un modèle de classification binaire.
