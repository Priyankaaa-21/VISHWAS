# Skin-Tone Label Agreement & Auditing Report

Analysis of inter-annotator agreement and objective CIELAB Individual Typology Angle (ITA) skin-tone grouping.

## 1. Group Mapping Definitions
- **Light**: Fitzpatrick I & II | ITA > 41°
- **Medium**: Fitzpatrick III & IV | 28° ≤ ITA ≤ 41°
- **Dark**: Fitzpatrick V & VI | ITA < 28°

**Total evaluated samples with valid annotations**: 144 / 168

## 2. Agreement: Fitzpatrick Label vs. Objective ITA Group
**Overall Group Agreement**: `66.7%`

```
Objective ITA Group  dark  light
Fitzpatrick Label               
dark                   48      0
light                   0     48
medium                 24     24
```

## 3. Inter-Annotator Agreement: Fitzpatrick (Ann 1) vs Centaur (Ann 2)
**Inter-Annotator Group Agreement**: `71.9%`

```
Centaur (Ann 2)      dark  light  medium
Fitzpatrick (Ann 1)                     
dark                   34      3      10
light                   0     45       2
medium                  2     22      21
```

## 4. Agreement: Centaur (Ann 2) vs Objective ITA Group
**Agreement**: `68.3%`

```
Objective ITA Group  dark  light
Centaur (Ann 2)                 
dark                   36      0
light                  11     59
medium                 23     10
```

## 5. Clinical & Fairness Insights
- Visual inspection of subjective Fitzpatrick scale annotations often suffers from significant inter-annotator variance.
- Objective CIELAB ITA offers a standardized, reproducible skin-tone metric that avoids subjective annotator drift.
- In downstream fairness auditing (`evaluate.py`), metrics are reported across **both** Fitzpatrick annotations and objective ITA groupings.
