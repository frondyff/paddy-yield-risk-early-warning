# Actionable Recommendations (Draft)

## 1) What to change (modifiable levers)
- Prioritize `Pest_60Day(in ml)` in agronomic planning and optimization tests.
- Prioritize `Seedrate(in Kg)` in agronomic planning and optimization tests.
- Prioritize `Micronutrients_70Days` in agronomic planning and optimization tests.
- Prioritize `Weed28D_thiobencarb` in agronomic planning and optimization tests.
- Prioritize `Potassh_50Days` in agronomic planning and optimization tests.
- Prioritize `LP_Mainfield(in Tonnes)` in agronomic planning and optimization tests.
- Prioritize `Variety` in agronomic planning and optimization tests.
- Prioritize `Nursery area (Cents)` in agronomic planning and optimization tests.

Use the following high-yield rule candidates as operational hypotheses (associative, not causal):
- `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) <= 5.500 AND Micronutrients_70Days > 67.500` (support 29.3%, pred lift +214.23)
- `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) <= 5.500 AND Micronutrients_70Days <= 67.500` (support 22.8%, pred lift +125.08)
- `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) > 5.500` (support 8.3%, pred lift +60.98)
- `Weed28D_thiobencarb <= 7.000 AND Nursery area (Cents) <= 30.000` (support 8.7%, pred lift -239.52)
- `Weed28D_thiobencarb <= 7.000 AND Nursery area (Cents) > 30.000 AND LP_nurseryarea(in Tonnes) > 2.500` (support 15.7%, pred lift -244.24)

## 2) What to control for (context features)
- Keep `Hectares` as a control/context variable in analysis and validation.
- Keep `Soil Types` as a control/context variable in analysis and validation.

## 3) What to avoid (proxy/leakage features)
- Exclude proxy/leakage variables from optimization and reporting metrics.
- Examples from current dictionary and audits:
- `30DRain( in mm)`
- `30DAI(in mm)`
- `30_50DRain( in mm)`
- `30_50DAI(in mm)`
- `51_70DRain(in mm)`
- `51_70AI(in mm)`
- `71_105DRain(in mm)`
- `71_105DAI(in mm)`
- `Min temp_D1_D30`
- `Max temp_D1_D30`
- `Min temp_D31_D60`
- `Max temp_D31_D60`

Reason: these variables can encode location or post-outcome information and inflate apparent performance.
All recommendations here are predictive-association based; causal conclusions require additional design.