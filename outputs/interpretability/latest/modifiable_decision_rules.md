# Modifiable-Only Decision Rules

Rules are extracted from a surrogate tree trained on model predictions using only numeric modifiable features.

- **R1**: `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) <= 5.500 AND Micronutrients_70Days > 67.500`
  Support: 686 rows (29.3%), Pred lift: +214.23, Actual lift: +218.09
- **R2**: `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) <= 5.500 AND Micronutrients_70Days <= 67.500`
  Support: 533 rows (22.8%), Pred lift: +125.08, Actual lift: +123.98
- **R3**: `Weed28D_thiobencarb > 7.000 AND LP_nurseryarea(in Tonnes) > 5.500`
  Support: 193 rows (8.3%), Pred lift: +60.98, Actual lift: +57.73
- **R4**: `Weed28D_thiobencarb <= 7.000 AND Nursery area (Cents) <= 30.000`
  Support: 203 rows (8.7%), Pred lift: -239.52, Actual lift: -239.31
- **R5**: `Weed28D_thiobencarb <= 7.000 AND Nursery area (Cents) > 30.000 AND LP_nurseryarea(in Tonnes) > 2.500`
  Support: 366 rows (15.7%), Pred lift: -244.24, Actual lift: -247.80