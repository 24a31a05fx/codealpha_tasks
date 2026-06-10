# Unemployment Analysis with Python
## Project Report — Internship Submission

**Author:** Data Analytics Internship Project  
**Dataset:** CMIE India Unemployment Data  
**Period Covered:** May 2019 – November 2020  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn  

---

## 1. Introduction

Unemployment is one of the most critical socioeconomic indicators, reflecting the health of an economy and the welfare of its population. In India, the COVID-19 pandemic of 2020 triggered an unprecedented shock to the labour market, with unemployment rates surging to historical highs within weeks of the nationwide lockdown announced in March 2020.

This project presents a comprehensive data-driven analysis of India's unemployment landscape, using real monthly state-level data from the Centre for Monitoring Indian Economy (CMIE). The analysis spans May 2019 to November 2020, encompassing the pre-pandemic baseline, the COVID-19 crisis period, and the early recovery phase.

---

## 2. Objective

The primary objectives of this project are:

- Understand India's unemployment landscape using real CMIE data
- Clean, preprocess, and validate the raw dataset for analysis
- Perform professional Exploratory Data Analysis (EDA)
- Quantify the economic shock caused by the COVID-19 pandemic
- Identify regional disparities and vulnerable states
- Analyse time-series trends, seasonal patterns, and moving averages
- Generate actionable, evidence-backed policy recommendations

---

## 3. Dataset Description

### 3.1 Source
Centre for Monitoring Indian Economy (CMIE) — Unemployment in India dataset.

### 3.2 Files

**File 1: `unemployment_india.csv`**
- **Records:** 740 (after cleaning)
- **Period:** May 2019 – June 2020
- **Granularity:** State × Area (Rural / Urban / Overall) × Month
- **Columns:** Region, Date, Frequency, Estimated Unemployment Rate (%), Estimated Employed, Estimated Labour Participation Rate (%), Area
- **States covered:** 28 states and Union Territories

**File 2: `unemployment_rate_2020.csv`**
- **Records:** 267
- **Period:** January 2020 – November 2020
- **Granularity:** State × Month
- **Columns:** Region, Date, Frequency, Estimated Unemployment Rate (%), Estimated Employed, Estimated Labour Participation Rate (%), Zone (North/South/East/West/Northeast), Longitude, Latitude

### 3.3 Key Variables

| Variable | Description | Type |
|----------|-------------|------|
| Estimated Unemployment Rate (%) | % of labour force unemployed | Float |
| Estimated Employed | Count of employed persons | Integer |
| Estimated Labour Participation Rate (%) | % of working-age population active | Float |
| Region | State / Union Territory | String |
| Date | Month-end date | Date |
| Area | Rural / Urban / Overall | String |
| Zone | Geographic zone (North, South, East, West, Northeast) | String |

### 3.4 Descriptive Statistics

| Statistic | Unemployment Rate (%) | Labour Participation Rate (%) |
|-----------|----------------------|-------------------------------|
| Mean | 11.79 (India) / 12.24 (2020) | 42.63 / 41.68 |
| Median | 8.35 / 9.65 | 41.16 / 40.39 |
| Std Dev | 10.72 / 10.80 | 8.11 / 7.85 |
| Min | 0.00 / 0.50 | 13.33 / 16.77 |
| Max | 76.74 / 75.85 | 72.57 / 69.69 |

The wide standard deviation (≈10.7%) indicates significant variation across states and time periods.

---

## 4. Methodology

The project follows a structured, reproducible analytics pipeline:

```
1. Data Extraction (ZIP → CSV)
2. Data Inspection (shape, types, nulls, duplicates)
3. Data Cleaning (standardise, parse, fill, validate)
4. Feature Engineering (date parts, period flags)
5. Exploratory Data Analysis
6. COVID-19 Impact Analysis
7. Time-Series & Trend Analysis
8. Correlation & Advanced Analysis
9. Insight Generation
10. Policy Recommendations
```

All code is written in Python, follows PEP8 standards, and is fully reproducible.

---

## 5. Data Cleaning

### 5.1 Steps Performed

| Step | Action | Reason |
|------|--------|--------|
| Column normalisation | Stripped spaces, lowercased, replaced special chars with `_` | Consistent programmatic access |
| Date parsing | `pd.to_datetime(format="%d-%m-%Y")` | Enables time-series operations |
| Null row removal | Dropped 28 fully-null rows from `df_india` | Blank separator rows in raw CSV |
| Duplicate removal | `drop_duplicates()` | 0 duplicates found in both files |
| Area null fill | Filled `NaN` → `"Overall"` | Contextually correct — rows without area tag represent national totals |
| Column rename | `region_1` → `zone` in df_2020 | Semantic clarity |
| Period flagging | Tagged each row as Pre-COVID / COVID / Recovery | Enables phase-wise comparative analysis |
| Temporal features | Extracted year, month, month name, quarter | Enables time-series aggregation |

### 5.2 Data Quality Assessment

- **Missing values:** 0 missing values in key numeric columns after cleaning
- **Outliers:** IQR analysis identified extreme values (unemployment > 65%) in states like Haryana and Tripura — these are genuine data points reflecting crisis conditions, not errors
- **Range validation:** All unemployment rates fall within [0, 100] — valid
- **Date continuity:** Monthly cadence maintained throughout

---

## 6. Exploratory Data Analysis (EDA)

### 6.1 Distribution Analysis

The unemployment rate follows a right-skewed distribution with:
- Mean: ~11.8%, Median: ~8.4%
- The skew is driven by COVID-period spikes in certain states
- KDE plots confirm bimodal behaviour: a "normal" cluster around 5–10% and a COVID spike cluster at 20–40%

### 6.2 Area-wise Unemployment

| Area | Mean Unemployment (%) | Interpretation |
|------|-----------------------|----------------|
| Rural | ~9.5% | Agricultural buffers absorbed some shock |
| Overall | ~11.8% | National composite |
| Urban | ~14.2% | Higher dependency on formal/service sector |

Urban areas showed higher and more volatile unemployment because the formal sector (offices, retail, transport) was more directly impacted by lockdown restrictions.

### 6.3 Outlier Analysis

Using the IQR method (1.5×IQR threshold):
- **~12% of records** were flagged as outliers
- Outliers predominantly correspond to April–June 2020 (COVID peak) in states like Haryana (76.7%), Jharkhand, and Tripura
- These are genuine economic events and were retained in the analysis

### 6.4 Correlation Findings

From the correlation matrix:
- **Unemployment Rate ↔ Labour Participation Rate: moderate negative correlation**  
  When labour participation drops (people stop looking for work), recorded unemployment also falls — a "discouraged worker" effect
- **Unemployment Rate ↔ Employed Persons: strong negative correlation**  
  Expected inverse relationship — fewer employed persons = higher unemployment

---

## 7. COVID-19 Impact Analysis

### 7.1 Phase Classification

| Phase | Period | Description |
|-------|--------|-------------|
| Pre-COVID | Jan–Feb 2020 | Baseline — pre-lockdown normal |
| COVID Peak | Mar–Jun 2020 | India's strictest nationwide lockdown (Mar 25) |
| Recovery | Jul–Nov 2020 | Phased unlock, gradual economic reopening |

### 7.2 National-Level Impact

| Metric | Value |
|--------|-------|
| Pre-COVID average unemployment | **9.23%** |
| COVID peak average unemployment | **16.74%** |
| Recovery average unemployment | **9.22%** |
| Absolute spike (percentage points) | **+7.51 pp** |
| Relative spike (%) | **+81.4%** |
| Recovery from peak | **-44.9%** |

**The COVID-19 pandemic caused an 81.4% increase in India's average unemployment rate in just 4–6 weeks.**

### 7.3 Monthly Peaks

- **April 2020** registered the highest single-month unemployment rate — aligned with the strictest phase of India's lockdown
- **May 2020** remained elevated as partial unlock began
- **October 2020** showed near-full national recovery to pre-COVID levels

### 7.4 State-Level COVID Impact

**Most affected states (largest relative spike):**
- **Puducherry** — extreme spike due to small base and tourism collapse
- **Haryana** — highest absolute rate, reaching 27.5% annually
- **Jharkhand** — manufacturing and mining disruption

**Most resilient states:**
- **Meghalaya** — lowest average unemployment (3.9%), agricultural economy
- **Northeast region overall** — subsistence agriculture provided natural buffer

### 7.5 Rural vs Urban Divergence

- **Urban unemployment spiked sharply** in March–April 2020 due to lockdown closure of offices, retail, construction, and transport
- **Rural unemployment rose** but more gradually — Kharif sowing season (June–July) provided employment continuity
- **Urban recovery was faster** post-unlock as the formal sector reopened

### 7.6 Zone-wise Analysis

| Zone | COVID Peak Avg (%) | Pre-COVID Avg (%) | Change |
|------|-------------------|-------------------|--------|
| North | ~22% | ~11% | +100% |
| South | ~18% | ~10% | +80% |
| West | ~20% | ~9% | +122% |
| East | ~16% | ~9% | +78% |
| Northeast | ~12% | ~7% | +71% |

The North and West zones experienced the largest relative spikes, likely due to their higher concentration of urban manufacturing and service sector workers.

---

## 8. Time-Series & Trend Analysis

### 8.1 Overall Trend (May 2019 – Nov 2020)

- **2019 (May–Dec):** Relatively stable unemployment in the 8–12% range
- **Jan–Feb 2020:** Slight uptick as economic slowdown effects lingered
- **Mar–May 2020:** Explosive rise — the COVID shock
- **Jun–Nov 2020:** Steady recovery trend

### 8.2 Moving Averages

- **3-Month Moving Average** smoothed short-term volatility and clearly showed the COVID spike
- **6-Month Moving Average** revealed the underlying trend: India's unemployment was already slightly elevated in early 2020 before COVID hit

### 8.3 Quarterly Analysis

| Quarter | Avg Unemployment (%) |
|---------|----------------------|
| Q2 2019 (May–Jun) | ~9.5% |
| Q3 2019 | ~10.1% |
| Q4 2019 | ~9.8% |
| Q1 2020 | ~10.2% |
| Q2 2020 | ~20.4% ← COVID peak |
| Q3 2020 | ~9.5% |
| Q4 2020 (Oct–Nov) | ~8.9% |

### 8.4 Seasonal Patterns

The data suggests a mild seasonal pattern:
- Unemployment tends to be **lower in Q1** (harvest/rabi season provides rural employment)
- **Higher in Q2–Q3** in non-COVID years (transition between agricultural cycles)
- This pattern was completely overwhelmed by the COVID shock in 2020

---

## 9. Key Findings

### Finding 1: COVID was an unprecedented labour market shock
An 81.4% spike in unemployment in 6 weeks has no historical parallel in modern India. The severity indicates how dependent millions of workers were on daily-wage and informal employment with no safety nets.

### Finding 2: Urban informal workers were most vulnerable
Urban areas showed sharper spikes. Informal urban workers — street vendors, construction labourers, domestic workers, auto-rickshaw drivers — had no income protection and are not fully captured by formal unemployment statistics.

### Finding 3: Recovery was remarkably rapid but uneven
By October 2020, the national average had returned to ~9.2% — close to the 9.23% pre-COVID baseline. However, the recovery was uneven: states like Haryana remained elevated while agricultural states like Meghalaya were barely affected.

### Finding 4: Labour participation dropped alongside unemployment
During COVID, both unemployment AND labour participation dropped simultaneously — a classic "discouraged worker" effect, where people stop seeking jobs entirely. This means actual labour market distress was likely worse than unemployment figures suggest.

### Finding 5: Northeast India showed structural resilience
The Northeast zone had consistently lower unemployment and smaller COVID spikes. The region's higher dependence on subsistence agriculture and smaller formal sector limited the lockdown's impact on registered unemployment.

### Finding 6: State-level disparities are stark
Even excluding COVID, the gap between Haryana (~27.5%) and Meghalaya (~3.9%) reflects massive structural differences in labour market composition, industrial base, and migration patterns.

---

## 10. Policy Recommendations

### Recommendation 1: Build a National Unemployment Safety Net
**Evidence:** 81.4% spike in weeks, with no income support for informal workers  
**Action:** India needs an urban equivalent of MGNREGS — a guaranteed minimum employment programme for urban informal workers activated automatically during economic shocks

### Recommendation 2: Invest in State-Specific Interventions for High-Risk States
**Evidence:** Haryana (27.5%), Jharkhand, and Tripura showed structurally high unemployment  
**Action:** Targeted industrial development, skills training, and MSE (Micro & Small Enterprises) promotion in these states

### Recommendation 3: Strengthen Labour Market Data Infrastructure
**Evidence:** Informal workers and discouraged workers are under-counted  
**Action:** Expand CMIE-style real-time tracking to district level; include gig economy workers and informal sector in official labour statistics

### Recommendation 4: Create a COVID-Economic Shock Preparedness Framework
**Evidence:** The 2020 crisis showed that lockdowns without economic support lead to unemployment spikes that disproportionately hurt the poor  
**Action:** Develop pre-approved economic stimulus protocols (direct cash transfers, MSE loans, rent moratoriums) that can be activated within 48 hours of a declared crisis

### Recommendation 5: Promote Rural-Urban Balanced Development
**Evidence:** Rural areas were more resilient due to agricultural employment buffers  
**Action:** Develop agro-processing industries in rural areas to create formal employment while maintaining agricultural fallback options

### Recommendation 6: Address Urban Youth Unemployment
**Evidence:** Urban unemployment was consistently higher and more volatile  
**Action:** Expand apprenticeship programmes, digital skilling initiatives, and start-up support to reduce youth dependency on fragile informal urban employment

---

## 11. Executive Summary

**Objective:** Analyse India's unemployment using CMIE state-level monthly data (2019–2020), with special focus on COVID-19 impact.

**Methodology:** End-to-end Python pipeline — data cleaning → EDA → COVID phase analysis → time-series analysis → 8 visualisation panels → insights.

**Key Findings:**
- India's unemployment rate surged **81.4%** (9.23% → 16.74%) during the COVID lockdown
- **Haryana** was the worst-affected state with an annual average of 27.5%
- **April–May 2020** marked the unemployment peak, coinciding with India's strictest lockdown
- **Urban workers** faced sharper and faster unemployment spikes than rural workers
- Recovery was largely complete by **October 2020** at the national level
- **Northeast India** showed structural resilience with smallest COVID-period spikes

**Recommendations:** Urban employment safety net, state-targeted interventions for high-risk states, real-time informal sector monitoring, and a pre-approved economic shock response framework.

**Conclusion:** This analysis demonstrates the power of data-driven policy evaluation. India's labour market response to COVID-19 exposed deep structural vulnerabilities — particularly among informal urban workers. Addressing these vulnerabilities requires both immediate social protection mechanisms and long-term structural reforms to create more resilient, formally-employed workforces.

---

*Report generated as part of Data Analytics Internship Project.*  
*Dataset: CMIE Unemployment in India | Tools: Python, Pandas, Matplotlib, Seaborn*
