# Research Overview

## Executive Summary
Short‑term thermal spikes—periods where land‑surface temperature rises 2–3 standard deviations above the 30‑year historical mean for 48–72 hours—can induce acute crop stress, especially during phenological stages such as flowering or grain filling. Coupled with precipitation deficits, these anomalies elevate the risk of yield loss.

## Analytical Formulas
### NDVI
$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
where **NIR** and **Red** are the surface reflectance values in the near‑infrared and red spectral bands, respectively.

### Temperature Anomaly Z‑Score
$$Z_{\text{temp}} = \frac{\text{LST}_{\text{obs}} - \mu_{\text{LST}}}{\sigma_{\text{LST}}}$$
* $\text{LST}_{\text{obs}}$ – observed land‑surface temperature.
* $\mu_{\text{LST}}$, $\sigma_{\text{LST}}$ – historical mean and standard deviation for the same day‑of‑year.

### Integrated Crop Stress Index (ICSI)
The ICSI combines vegetation health, thermal stress, and moisture deficit into a single metric:
$$\text{ICSI} = w_{\text{NDVI}} \cdot (1 - \text{NDVI}) + w_{\text{temp}} \cdot \max(0, Z_{\text{temp}}) + w_{\text{prec}} \cdot \Delta\text{P}\$$
* $w_{\text{NDVI}}, w_{\text{temp}}, w_{\text{prec}}$ – weighting coefficients (default 0.4, 0.35, 0.25).
* $\Delta\text{P}$ – precipitation deficit (mm) relative to the climatological norm for the growth window.

## Risk Indicator Table
| Stress Tier | $Z_{\text{temp}}$ Threshold | $\Delta\text{NDVI}$ (ΔNDVI) | $\Delta\text{P}$ (mm deficit) | Typical Impact |
|------------|----------------------------|-----------------------------|--------------------------------|----------------|
| Normal     | < 1.0                      | < 0.02                     | < 5                            | No visible stress |
| Watch      | 1.0 – 1.5                  | 0.02 – 0.05                | 5 – 15                         | Minor chlorosis |
| Warning    | 1.5 – 2.0                  | 0.05 – 0.10                | 15 – 30                        | Reduced leaf area |
| Critical   | > 2.0                      | > 0.10                     | > 30                           | Wilting, yield loss |

## Future Roadmap
- **Synthetic Aperture Radar (SAR) integration** – Incorporate moisture‑sensitive radar backscatter to improve drought detection under cloud cover.
- **Phenology‑aware weighting** – Dynamically adjust $w_{\text{temp}}$ and $w_{\text{prec}}$ based on growth stage using vegetation phenology curves.
- **Machine‑learning yield prediction** – Train regression models on historic yield data using ICSI and ancillary variables (soil type, management practices).

The methodology outlined here provides a reproducible framework for early‑warning agro‑environmental monitoring.
