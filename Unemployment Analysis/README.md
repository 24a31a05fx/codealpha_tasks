# 📊 Unemployment Analysis with Python
### India | CMIE Dataset | 2019–2020

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-green)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Overview

A professional end-to-end data analytics project analyzing unemployment trends in India from May 2019 to November 2020. The project investigates the devastating impact of the COVID-19 pandemic on India's labour market, identifies regional patterns, and provides evidence-based policy recommendations.

This project was developed as an internship submission and is structured for GitHub portfolio publication.

---

## 📁 Dataset Description

Two datasets sourced from the **Centre for Monitoring Indian Economy (CMIE)**:

| File | Rows | Period | Granularity |
|------|------|--------|-------------|
| `unemployment_india.csv` | 740 | May 2019 – Jun 2020 | State × Area (Rural/Urban/Overall) |
| `unemployment_rate_2020.csv` | 267 | Jan 2020 – Nov 2020 | State × Month with geo-coordinates |

### Key Variables
- **Estimated Unemployment Rate (%)** — Primary target variable
- **Estimated Employed** — Number of employed persons
- **Estimated Labour Participation Rate (%)** — % of working-age population active in labour force
- **Region / Area / Zone** — Geographic dimensions (28 states, 5 zones)

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Core language |
| Pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical computing |
| Matplotlib | 3.7+ | Visualizations |
| Seaborn | 0.12+ | Statistical plots |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Unemployment-Analysis.git
cd Unemployment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python src/analysis.py
```

---

## 🗂️ Project Structure

```
Unemployment-Analysis/
│
├── data/
│   ├── unemployment_india.csv          # CMIE India dataset (Rural/Urban/Overall)
│   └── unemployment_rate_2020.csv      # 2020 state-level dataset with geo-coords
│
├── images/
│   ├── 01_distribution_outlier.png     # Distribution & outlier charts
│   ├── 02_national_trend_covid.png     # National trend + COVID analysis
│   ├── 03_statewise_analysis.png       # All-state unemployment rankings
│   ├── 04_zone_analysis.png            # 5-zone trend comparison
│   ├── 05_covid_impact.png             # COVID impact deep-dive
│   ├── 06_trend_analysis.png           # Time-series & seasonal analysis
│   ├── 07_correlation_advanced.png     # Correlation & advanced insights
│   └── 08_executive_dashboard.png      # Executive summary dashboard
│
├── src/
│   └── analysis.py                     # Main analysis script (PEP8, modular)
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── report.md                           # Full internship submission report
```

---

## 🔄 Project Workflow

```
Data Loading → Cleaning → EDA → COVID Analysis → Trend Analysis → Visualizations → Insights → Recommendations
```

1. **Data Loading** — Two CSVs extracted, inspected for shape, types, nulls
2. **Cleaning** — Column standardisation, date parsing, duplicate removal, null handling
3. **EDA** — Distributions, boxplots, IQR outliers, correlation matrix
4. **COVID Analysis** — Pre/During/Post phase comparison, state-level spike quantification
5. **Trend Analysis** — Monthly/quarterly/seasonal patterns, rolling averages
6. **Visualizations** — 8 professional chart panels, 20+ individual charts
7. **Insights & Recommendations** — Data-driven, evidence-linked findings

---

## 📊 Analysis Performed

- **Distribution Analysis** — Histograms, KDE plots, density overlays
- **Outlier Detection** — IQR method identified outliers in unemployment extremes
- **Geographic Analysis** — 27 states, 5 zones (North/South/East/West/Northeast)
- **COVID-19 Impact** — Quantified spike, peak, and recovery with statistical evidence
- **Time-Series Analysis** — 3-month and 6-month moving averages
- **Rural vs Urban** — Comparative unemployment across area types
- **Correlation Analysis** — Unemployment vs Labour Participation Rate relationship

---

## 📈 Key Findings

| Metric | Value |
|--------|-------|
| Pre-COVID avg unemployment | 9.23% |
| COVID peak avg unemployment | 16.74% |
| Recovery avg unemployment | 9.22% |
| Spike from Pre→COVID | **+81.4%** |
| Recovery from peak | -44.9% |
| Highest state (2020) | Haryana (27.5%) |
| Lowest state (2020) | Meghalaya (3.9%) |
| States analysed | 27 |

### Top Insights
1. **COVID caused an 81.4% spike** in unemployment — from 9.23% to 16.74% in just weeks
2. **Urban areas were hit harder** than rural areas during lockdown (demand collapse)
3. **Northeast India was most resilient** — lowest COVID-period unemployment spikes
4. **April–May 2020 was the peak** — coinciding with India's strictest nationwide lockdown
5. **Recovery was rapid but incomplete** — unemployment returned to pre-COVID levels by Oct 2020 for most states

---

## 🔭 Future Scope

- Extend dataset to 2021–2023 to analyse long-term post-COVID recovery
- Incorporate sector-wise data (agriculture, manufacturing, services)
- Build a time-series forecasting model (ARIMA / Prophet)
- Add interactive dashboards using Plotly / Dash
- Integrate GDP and inflation data for macro-economic correlation analysis
- State-level choropleth map visualisation

---

## 📝 Author

Data Analytics Internship Project — Built with CMIE India unemployment data.

---

*This project is suitable for academic submission, internship evaluation, and GitHub portfolio publication.*
