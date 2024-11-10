**Équations des métriques de performance :**

---
**Légende :**

- $\text{VP}$ : Vrais Positifs (True Positives)
- $\text{VN}$ : Vrais Négatifs (True Negatives)
- $\text{FP}$ : Faux Positifs (False Positives)
- $\text{FN}$ : Faux Négatifs (False Negatives)

----

**1. Sensibilité (Recall, Taux de Vrais Positifs, TPR)**

$$
\text{Sensibilité} = \text{TPR} = \frac{\text{VP}}{\text{VP} + \text{FN}}
$$

---

**2. Spécificité (Taux de Vrais Négatifs, TNR)**

$$
\text{Spécificité} = \text{TNR} = \frac{\text{VN}}{\text{VN} + \text{FP}}
$$

---

**3. Précision (Valeur Prédictive Positive, PPV)**

$$
\text{Précision} = \text{PPV} = \frac{\text{VP}}{\text{VP} + \text{FP}}
$$

---

**4. Valeur Prédictive Négative (NPV)**

$$
\text{Valeur\ Prédictive\ Négative} = \text{NPV} = \frac{\text{VN}}{\text{VN} + \text{FN}}
$$

---

**5. Taux de Faux Positifs (FPR)**

$$
\text{Taux\ de\ Faux\ Positifs} = \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{VN}}
$$

---

**6. Taux de Faux Négatifs (FNR)**

$$
\text{Taux\ de\ Faux\ Négatifs} = \text{FNR} = \frac{\text{FN}}{\text{FN} + \text{VP}}
$$

---

**7. Taux de Faux Découvertes (FDR)**

$$
\text{Taux\ de\ Faux\ Découvertes} = \text{FDR} = \frac{\text{FP}}{\text{FP} + \text{VP}}
$$

---

**8. Précision Globale (Exactitude, ACC)**

$$
\text{Précision\ Globale} = \text{ACC} = \frac{\text{VP} + \text{VN}}{\text{VP} + \text{VN} + \text{FP} + \text{FN}}
$$

---

**9. Score F1**

$$
F1 = 2 \times \frac{\text{Précision} \times \text{Sensibilité}}{\text{Précision} + \text{Sensibilité}}
$$

---

**10. Coefficient de Corrélation de Matthews (MCC)**

$$
\text{MCC} = \frac{ (\text{VP} \times \text{VN}) - (\text{FP} \times \text{FN}) }{ \sqrt{ (\text{VP} + \text{FP})(\text{VP} + \text{FN})(\text{VN} + \text{FP})(\text{VN} + \text{FN}) } }
$$

---

**Légende :**

- $\text{VP}$ : Vrais Positifs (True Positives)
- $\text{VN}$ : Vrais Négatifs (True Negatives)
- $\text{FP}$ : Faux Positifs (False Positives)
- $\text{FN}$ : Faux Négatifs (False Negatives)
