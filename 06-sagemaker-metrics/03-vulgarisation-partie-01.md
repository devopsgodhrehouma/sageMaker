**Comprendre les métriques de performance en classification binaire :**

# Référence :
- https://en.wikipedia.org/wiki/Precision_and_recall
  
Lorsqu'on évalue un modèle de classification binaire (par exemple, prédire si un patient est "normal" ou "anormal"), il est essentiel de comprendre les différentes métriques qui mesurent sa performance. Voici une explication simplifiée des principales métriques, avec des exemples pour faciliter la compréhension.

---

### **1. Sensibilité (Recall ou Taux de Vrais Positifs)**

La **sensibilité** mesure la capacité du modèle à identifier correctement les cas positifs réels.

- **En termes simples** : Parmi tous les patients réellement anormaux, combien le modèle en a-t-il correctement identifiés comme anormaux ?

- **Formule** :

  $$
  \text{Sensibilité} = \frac{\text{Vrais Positifs (VP)}}{\text{Vrais Positifs (VP)} + \text{Faux Négatifs (FN)}}
  $$

- **Exemple** : Si sur 100 patients anormaux, le modèle en identifie correctement 90, la sensibilité est de 90%.

- **Importance** : Une haute sensibilité est cruciale lorsque manquer un cas positif peut avoir de graves conséquences, comme dans le dépistage de maladies.

---

### **2. Spécificité (Taux de Vrais Négatifs)**

La **spécificité** mesure la capacité du modèle à identifier correctement les cas négatifs réels.

- **En termes simples** : Parmi tous les patients réellement normaux, combien le modèle en a-t-il correctement identifiés comme normaux ?

- **Formule** :

  $$
  \text{Spécificité} = \frac{\text{Vrais Négatifs (VN)}}{\text{Vrais Négatifs (VN)} + \text{Faux Positifs (FP)}}
  $$

- **Exemple** : Si sur 100 patients normaux, le modèle en identifie correctement 80, la spécificité est de 80%.

- **Importance** : Une haute spécificité réduit les faux positifs, évitant des tests supplémentaires ou de l'anxiété inutile.

---

### **3. Précision (Valeur Prédictive Positive)**

La **précision** indique la proportion de prédictions positives qui sont correctes.

- **En termes simples** : Parmi tous les patients que le modèle a prédits comme anormaux, combien sont réellement anormaux ?

- **Formule** :

  $$
  \text{Précision} = \frac{\text{Vrais Positifs (VP)}}{\text{Vrais Positifs (VP)} + \text{Faux Positifs (FP)}}
  $$

- **Exemple** : Si le modèle prédit 50 patients comme anormaux et que 45 le sont réellement, la précision est de 90%.

- **Importance** : Une haute précision signifie que les prédictions positives sont fiables.

---

### **4. Valeur Prédictive Négative (VPN)**

La **VPN** mesure la proportion de prédictions négatives qui sont correctes.

- **En termes simples** : Parmi tous les patients que le modèle a prédits comme normaux, combien sont réellement normaux ?

- **Formule** :

  $$
  \text{VPN} = \frac{\text{Vrais Négatifs (VN)}}{\text{Vrais Négatifs (VN)} + \text{Faux Négatifs (FN)}}
  $$

- **Exemple** : Si le modèle prédit 70 patients comme normaux et que 65 le sont réellement, la VPN est d'environ 92.86%.

- **Importance** : Une haute VPN rassure sur le fait que les prédictions négatives sont fiables.

---

### **5. Taux de Faux Positifs (FPR)**

Le **FPR** mesure la proportion de cas négatifs réels mal classés comme positifs.

- **En termes simples** : Parmi tous les patients réellement normaux, combien le modèle a-t-il incorrectement identifiés comme anormaux ?

- **Formule** :

  $$
  \text{FPR} = \frac{\text{Faux Positifs (FP)}}{\text{Vrais Négatifs (VN)} + \text{Faux Positifs (FP)}}
  $$

- **Exemple** : Si sur 100 patients normaux, 20 sont incorrectement prédits comme anormaux, le FPR est de 20%.

- **Importance** : Un FPR élevé peut entraîner des tests ou des traitements inutiles.

---

### **6. Taux de Faux Négatifs (FNR)**

Le **FNR** mesure la proportion de cas positifs réels mal classés comme négatifs.

- **En termes simples** : Parmi tous les patients réellement anormaux, combien le modèle a-t-il manqués en les identifiant comme normaux ?

- **Formule** :

  $$
  \text{FNR} = \frac{\text{Faux Négatifs (FN)}}{\text{Vrais Positifs (VP)} + \text{Faux Négatifs (FN)}}
  $$

- **Exemple** : Si sur 100 patients anormaux, 10 sont incorrectement prédits comme normaux, le FNR est de 10%.

- **Importance** : Un FNR élevé signifie que le modèle manque des cas positifs, ce qui peut être dangereux en médecine.

---

### **7. Taux de Faux Découvertes (FDR)**

Le **FDR** indique la proportion de prédictions positives qui sont incorrectes.

- **En termes simples** : Parmi tous les patients que le modèle a prédits comme anormaux, combien ne le sont pas réellement ?

- **Formule** :

  $$
  \text{FDR} = \frac{\text{Faux Positifs (FP)}}{\text{Vrais Positifs (VP)} + \text{Faux Positifs (FP)}}
  $$

- **Exemple** : Si le modèle prédit 50 patients comme anormaux et que 10 ne le sont pas, le FDR est de 20%.

- **Importance** : Un FDR élevé peut réduire la confiance dans les prédictions positives du modèle.

---

### **8. Précision Globale (Exactitude)**

La **précision globale** mesure le pourcentage total de prédictions correctes.

- **En termes simples** : Sur tous les patients, quelle proportion le modèle a-t-il correctement classés ?

- **Formule** :

  $$
  \text{Précision Globale} = \frac{\text{Vrais Positifs (VP)} + \text{Vrais Négatifs (VN)}}{\text{Total des cas}}
  $$

- **Exemple** : Si sur 200 patients, le modèle en classe correctement 170, la précision globale est de 85%.

- **Importance** : Bien qu'elle donne une vue d'ensemble, cette métrique peut être trompeuse si les classes sont déséquilibrées.

---

### **Illustration avec une Matrice de Confusion**

Voici comment ces métriques se rapportent à la matrice de confusion :

|                | **Prédit : Anormal (1)** | **Prédit : Normal (0)** |
|----------------|--------------------------|-------------------------|
| **Réel : Anormal (1)** | Vrais Positifs (VP)      | Faux Négatifs (FN)      |
| **Réel : Normal (0)**  | Faux Positifs (FP)       | Vrais Négatifs (VN)     |

---

### **Pourquoi ces métriques sont importantes ?**

- **Sensibilité élevée** : Assure que la plupart des cas positifs sont détectés. Crucial pour ne pas manquer des maladies.

- **Spécificité élevée** : Réduit les faux positifs, évitant des interventions inutiles.

- **Équilibre** : Souvent, il faut trouver un équilibre entre sensibilité et spécificité en fonction du contexte et des conséquences des erreurs.

---

### **Exemple Pratique**

Supposons que nous testons un modèle sur 100 patients :

- **50 patients anormaux** : Le modèle en identifie correctement 45 (VP) et en manque 5 (FN).
- **50 patients normaux** : Le modèle en identifie correctement 40 (VN) et en classe incorrectement 10 comme anormaux (FP).

Calculons les métriques :

- **Sensibilité** :

  $$
  \text{Sensibilité} = \frac{45}{45 + 5} = 0.9 \text{ ou } 90\%
  $$

- **Spécificité** :

  $$
  \text{Spécificité} = \frac{40}{40 + 10} = 0.8 \text{ ou } 80\%
  $$

- **Précision** :

  $$
  \text{Précision} = \frac{45}{45 + 10} \approx 0.818 \text{ ou } 81.8\%
  $$

- **Précision Globale** :

  $$
  \text{Précision Globale} = \frac{45 + 40}{100} = 0.85 \text{ ou } 85\%
  $$

---

### **Interprétation des Résultats**

- **Sensibilité de 90%** : Le modèle détecte correctement 90% des patients anormaux.

- **Spécificité de 80%** : Le modèle identifie correctement 80% des patients normaux.

- **Précision de 81.8%** : Lorsqu'il prédit une anomalie, il a raison environ 82% du temps.

- **Implications** :

  - **Faux Négatifs (FN)** : 5 patients anormaux non détectés. Cela peut être critique si la condition médicale nécessite une intervention rapide.
  
  - **Faux Positifs (FP)** : 10 patients normaux identifiés à tort comme anormaux, pouvant entraîner du stress ou des tests inutiles.

---

### **Conclusion**

Comprendre ces métriques permet de :

- **Évaluer la performance globale du modèle**.
- **Identifier les domaines d'amélioration** (par exemple, réduire les faux négatifs).
- **Adapter le modèle en fonction des priorités** (par exemple, maximiser la sensibilité dans le dépistage médical).

---

**En résumé**, ces métriques fournissent des informations détaillées sur la façon dont le modèle effectue des prédictions et les types d'erreurs qu'il commet. Cela aide à prendre des décisions éclairées sur l'utilisation du modèle dans des situations réelles et à apporter des ajustements pour améliorer ses performances.
