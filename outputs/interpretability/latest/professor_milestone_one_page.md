# Milestone 1-Page Summary

## Model and Evaluation
- Final candidate: `catboost` on `full_review` feature set.
- Primary validation (LOGO): R^2=0.5063, RMSE=202.69, MAE=159.56.
- Secondary validation (GroupShuffle): R^2=0.5077, RMSE=203.04, MAE=159.75.

## Interpretability Outputs Delivered
- Global SHAP importance generated and ranked by role (modifiable/context/proxy).
- Local SHAP examples generated for high/low prediction and major error cases.
- Surrogate decision rules extracted using only numeric modifiable features.

## Top Modifiable Levers (Current Model)
- `Pest_60Day(in ml)`
- `Seedrate(in Kg)`
- `Micronutrients_70Days`
- `Weed28D_thiobencarb`
- `Potassh_50Days`

## Rule-Based Operational Hypotheses
- `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) <= 5.500 AND Micronutrients_70Days > 67.500` (support 29.3%, actual lift +218.09)
- `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) <= 5.500 AND Micronutrients_70Days <= 67.500` (support 22.8%, actual lift +123.98)
- `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) > 5.500` (support 8.3%, actual lift +57.73)
- `Weed28D_thiobencarb <= 7.000 AND Nursery area (Cents) <= 30.000` (support 8.7%, actual lift -239.31)
- `Weed28D_thiobencarb <= 7.000 AND Nursery area (Cents) > 30.000 AND LP_nurseryarea(in Tonnes) > 2.500` (support 15.7%, actual lift -247.80)

## Limitations
- No explicit date/season variable is available, limiting temporal alignment and causal interpretation.
- Agriblock-linked confounding remains possible; results are associative, not causal.
- Findings should be validated through controlled field experiments or time-aware data collection.