# EnergyMind Phase 1

Détection et classification d'anomalies sur un micro-réseau électrique
intelligent : 20 maisons, 4 zones, 1 transformateur, mesures toutes les
15 minutes sur 240 jours.

## 1. Vue d'ensemble

Le système reçoit les relevés de compteurs et produit, pour chaque maison et
chaque pas de temps, un verdict parmi onze classes — fonctionnement normal ou
l'une des dix anomalies — accompagné d'une confiance calibrée.

Trois composants aux rôles étanches :

| Composant | Rôle | Ce qu'il ne fait pas |
|---|---|---|
| **LightGBM** | classe les anomalies connues | — |
| **Isolation Forest** | score d'atypicité / nouveauté | il ne classe rien, n'invente aucun nom de classe |
| **Règles physiques** | cohérence maison ↔ voisins ↔ zone ↔ transformateur | il n'apprend rien |

Principe directeur du moteur de règles : **aucun capteur n'est une vérité
absolue**. On compare trois agrégats mutuellement indépendants — la somme des
compteurs de maison, le compteur de zone, le compteur du transformateur — et
c'est leur désaccord qui désigne le menteur.

## 2. Architecture finale

```
mesures brutes
      │
      ▼
assemblage maison + zone + transformateur
      │
      ▼
   24 features
      │
      ├──────────────┬────────────────────┐
      ▼              ▼                    ▼
  LightGBM     Isolation Forest     règles physiques
      │              │                    │
      ▼              │                    │
 Vector Scaling      │                    │
      │              │                    │
      └──────────────┴────────────────────┘
                     │
                   fusion
                     │
                     ▼
        verdict + confiance calibrée
                     │
          confiance ≥ 0,80 ?  ──non──▶  INCERTAIN
                     │ oui
                     ▼
              diagnostic de classe
```

**Logique de fusion**, volontairement simple et documentée :

- LightGBM dit « normal » mais les preuves signalent une anomalie, et sa
  meilleure classe anormale dépasse `promote_min_prob` → promotion ;
- LightGBM dit « anormal » sans preuve et avec une confiance inférieure à
  `demote_max_prob` → retour à « normal ».

Les deux seuils sont choisis par recherche exhaustive **sur la validation
uniquement**.

## 3. Dataset

Jeu figé de 240 jours produit par le simulateur EnergyMind (projet séparé).

| Fichier | Lignes | Contenu |
|---|---|---|
| `data/raw_houses.csv` | 460 800 | `timestamp, house_id, zone_id, V, I, P, PF, E` |
| `data/raw_zones.csv` | 92 160 | `timestamp, zone_id, zone_P, zone_V` |
| `data/raw_transformer.csv` | 23 040 | `timestamp, trans_P` |
| `data/labels.csv` | 576 000 | cible supervisée, trois niveaux d'entités |

`ground_truth.csv` et `injection_log.csv` ne sont **jamais lus**. Le chargeur
refuse tout fichier dont le schéma comporte une colonne inattendue, ce qui
bloquerait une fuite de vérité de simulation.

`labels.csv` fournit `label` (la cible) et `event_id`. **`event_id` sert
exclusivement** à construire les découpages sans fuite et à l'évaluation au
niveau événement ; il n'entre jamais dans la matrice de features.

### Découpages temporels

Jamais de mélange aléatoire des lignes. Un découpage principal plus quatre
découpages glissants, pour mesurer la stabilité :

| Nom | Train | Validation | Test |
|---|---|---|---|
| PRIMARY | j1–160 | j161–200 | j201–240 |
| S1 | j1–120 | j121–140 | j141–180 |
| S2 | j1–140 | j141–160 | j161–200 |
| S3 | j1–160 | j161–180 | j181–220 |
| S4 | j1–180 | j181–200 | j201–240 |

Si un événement chevauche une frontière, **toutes ses lignes sont écartées** de
ce découpage : le même événement ne peut jamais apparaître des deux côtés.

## 4. Les 24 features

Toutes dérivées des seules mesures observables, toutes causales (pas courant et
pas passés uniquement).

**12 features d'origine** — `src/features/base_features.py`

| Feature | Mesure |
|---|---|
| `res_zone` | incohérence maisons ↔ compteur de zone, **perte technique déduite** |
| `res_trans` | incohérence zones ↔ compteur du poste, perte déduite |
| `res_energy` | `(E(t) − E(t−1)) − P × 0,25`, jamais l'index brut |
| `ratio_voisins` | la maison face aux autres de sa zone, après normalisation |
| `nb_voisins_anormaux` | nombre de voisins déviant au même instant |
| `variation` | `ΔP` normalisée par la baseline de la maison |
| `frozen_count` | paliers consécutifs quasi identiques, **avec tolérance** |
| `rolling_std_6h` | variabilité des 6 heures passées |
| `drift_slope` | pente de dérive sur 12 h, mesurée sur l'écart aux voisins |
| `dev_tension` | `(V − 230) / 230` |
| `trans_load_pct` | charge du transformateur en % du nominal |
| `res_VI` | cohérence `P ≈ V × I × PF / 1000` |

**4 écarts à l'historique de la maison** — `src/features/baseline_features.py`

`deviation_from_house_baseline`, `relative_deviation_from_baseline`,
`zscore_vs_house_baseline`, `ratio_to_hour_median`.

Elles comparent la maison à **sa propre histoire** et non à ses voisins : c'est
ce qui distingue un décalage de compteur d'une variation de zone.

**8 features de persistance** — `src/features/persistence_features.py`

`mean_deviation_3h/6h/12h/24h`, `median_res_zone_6h/24h`,
`persistent_neighbor_deviation`, `consecutive_abnormal_count`.

Un démarrage d'appareil dure 1 à 4 pas, un biais de compteur 24 à 72 : moyenner
sur plusieurs heures efface le premier et conserve le second.

### Statistiques apprises

| Statistique | Origine |
|---|---|
| baselines par maison / créneau / heure | médianes du **train** |
| échelles robustes (médiane, MAD) | **train** |
| modèle de perte technique (affine en charge) | régression sur le **train**, déciles 10–90 |
| tolérance `frozen_count` | quantile 2 % des variations relatives du **train** |
| seuil « voisin anormal » | quantile 99 % du **train** |
| seuils des règles | quantile 99,5 % sur les instants **propres** du train |

Aucune constante arbitraire. Le constructeur de features est **ré-ajusté pour
chaque découpage**.

### Valeurs manquantes

Les NaN sont **conservés** : ils portent de l'information (classe
`missing_data`) et LightGBM les traite nativement. Isolation Forest, qui ne les
accepte pas, reçoit une copie imputée par les **médianes du train**.

## 5. Structure du projet

```
EnergyMind-Phase1/
├── data/                        dataset figé de 240 jours
├── models/                      modèle entraîné (joblib)
├── src/
│   ├── config.py                chargement de config.yaml
│   ├── data/
│   │   ├── loader.py            chargement + inspection des CSV
│   │   ├── merge.py             assemblage maison + zone + transfo
│   │   └── splits.py            découpages, protection anti-fuite
│   ├── features/
│   │   ├── windows.py           primitives de fenêtre causale
│   │   ├── base_features.py     les 12 d'origine
│   │   ├── baseline_features.py les 4 d'historique
│   │   ├── persistence_features.py  les 8 de persistance
│   │   └── feature_pipeline.py  FeatureBuilder, FINAL_FEATURES
│   ├── models/
│   │   ├── lightgbm_model.py    classification multiclasse
│   │   ├── isolation_forest.py  score de nouveauté
│   │   └── calibration.py       Vector Scaling, ECE, abstention
│   ├── rules/physical_rules.py  cohérence entre niveaux
│   ├── inference/pipeline.py    préparation, fusion, politique de confiance
│   └── evaluation/
│       ├── metrics.py           métriques ligne et événement
│       └── plots.py             figures
├── outputs/{metrics,figures,predictions,datasets}/
├── tests/
├── config.yaml                  tous les paramètres
├── main.py                      entraînement + évaluation
├── run_calibration.py           calibration de confiance
└── requirements.txt
```

## 6. Installation

```bash
pip install -r requirements.txt
```

> **Note d'environnement.** `scikit-learn` est épinglé en **1.5.2** : sur
> certains postes Windows, la version 1.9.x est bloquée par la stratégie de
> contrôle d'applications (`DLL load failed ... _expected_mutual_info_fast`),
> ce qui rend `sklearn.ensemble` inutilisable.

## 7. Exécution

```bash
python main.py
```

Entraîne et évalue le modèle sur les 5 découpages (~45 min), écrit métriques,
figures, prédictions et le modèle sauvegardé.

```bash
python main.py --quick --no-figures
```

Découpage principal seul, pour une mise au point rapide.

```bash
python run_calibration.py
```

Calibration de confiance. Avec `--reuse`, réutilise les probabilités déjà
sauvegardées au lieu de réentraîner.

```bash
python -m pytest tests -q
```

27 tests : chargement, assemblage, 24 features, rôles des composants,
calibration, politique de confiance.

## 8. Calibration de confiance

Quatre méthodes sont comparées — aucune, temperature, vector, isotone — toutes
ajustées **sur la validation uniquement**, et sélectionnées par
log-vraisemblance de validation.

**Vector Scaling** est retenue. Effet mesuré sur les lignes prédites anormales :

| | Avant | Après |
|---|---|---|
| ECE | 0,1040 | **0,0453** |
| Précision réelle | 84,7 % | **90,6 %** |

Exemple parlant : le système annonçait `sensor_drift` avec 77 % de confiance en
n'ayant raison que 37 % du temps. Il annonce désormais 66 % pour 54 % de
réussite.

**Politique d'abstention** : au-dessous de `inference.confidence_threshold`
(0,80), le système répond **INCERTAIN** au lieu d'un nom de classe. On conserve
alors 89 % des alertes avec **96 % de précision**.

## 9. Résultats

Moyenne ± écart-type sur les 5 découpages temporels.

| Métrique | Valeur |
|---|---|
| Macro F1 | 0,830 ± 0,009 |
| Macro Recall | 0,830 |
| Rappel d'anomalies | 0,804 |
| **Taux de faux positifs** | **0,60 %** |
| Rappel au niveau événement | 98,5 % |
| Accuracy | 97,2 % |

L'accuracy est volontairement secondaire : 92 % des lignes sont normales, donc
un modèle muet afficherait déjà 92 %.

**F1 par classe** : `house_outage` et `zone_outage` 1,00 · `missing_data` 0,98 ·
`voltage_sag` 0,96 · `sensor_frozen` 0,96 · `overload` 0,97 ·
`voltage_spike` 0,91 · `falsified_reading` 0,82 · `sensor_bias` 0,33 ·
`sensor_drift` 0,21.

## 10. Limites connues

- **`sensor_bias` et `sensor_drift` restent faibles.** Le système détecte le
  problème dans 67 à 77 % des cas mais confond souvent les deux classes entre
  elles : elles se recouvrent réellement. Quatre familles de features ont été
  testées ; seules deux ont tenu sur les 5 découpages, et aucune n'a amélioré
  la dérive.
- **`sensor_bias` reste 27 points trop confiant** après calibration. Celle-ci
  rend l'optimisme visible, elle ne le supprime pas.
- **Retard de détection** : une dérive est détectée au bout de 26 pas en
  médiane (6 h 30), et jusqu'à 57 pas au 90ᵉ centile.
- **Un seul jeu simulé.** Les conclusions valent pour ce réseau et ce
  simulateur ; un réseau réel demandera une revalidation.
- L'écart entre F1 d'entraînement (0,99) et de test (0,83) est important : le
  modèle apprend presque parfaitement le train. Il ne se creuse pas avec les
  features ajoutées, mais il reste à surveiller.

## 11. Reproductibilité

- Graine fixée (`model.random_state: 42`), propagée à LightGBM et à
  Isolation Forest.
- Aucun chemin absolu dans les modules : la racine du projet est déduite de
  l'emplacement de `src/config.py`.
- Tous les paramètres sont dans `config.yaml`.
- Le modèle entraîné est sauvegardé dans `models/`, les probabilités de
  calibration dans `outputs/datasets/` pour rejouer l'analyse sans
  réentraînement.
