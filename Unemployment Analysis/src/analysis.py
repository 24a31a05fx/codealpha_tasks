"""
Unemployment Analysis with Python
==================================
Author: Data Analytics Project
Dataset: CMIE Unemployment Data – India (May 2019 – Nov 2020)
Description: End-to-end analysis of unemployment trends in India,
             including COVID-19 impact, regional patterns, and policy insights.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ── Global Style ──────────────────────────────────────────────────────────────
BG_COLOR   = "#F8F9FA"
ACCENT     = "#2563EB"
RED        = "#DC2626"
GREEN      = "#16A34A"
ORANGE     = "#EA580C"
DARK       = "#1E293B"
LIGHT_GRAY = "#E2E8F0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "images"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#CBD5E1",
    "axes.labelcolor":  DARK,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.titlepad":    12,
    "xtick.color":      DARK,
    "ytick.color":      DARK,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "grid.color":       LIGHT_GRAY,
    "grid.linewidth":   0.8,
    "legend.framealpha": 0.9,
    "legend.fontsize":  9,
    "font.family":      "DejaVu Sans",
})

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Saved: {name}")

print("=" * 60)
print("  UNEMPLOYMENT ANALYSIS WITH PYTHON")
print("  India | CMIE Dataset | 2019–2020")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/9] Loading and cleaning data…")

df_india = pd.read_csv(DATA_DIR / "unemployment_india.csv")
df_2020  = pd.read_csv(DATA_DIR / "unemployment_rate_2020.csv")

# Standardise column names
def clean_cols(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_")
    )
    return df

df_india = clean_cols(df_india)
df_2020  = clean_cols(df_2020)

# After cleaning:
# df_india : region | date | frequency | estimated_unemployment_rate
#             estimated_employed | estimated_labour_participation_rate | area
# df_2020  : region | date | frequency | estimated_unemployment_rate
#             estimated_employed | estimated_labour_participation_rate
#             region_1 | longitude | latitude

# Short aliases
UR  = "estimated_unemployment_rate"
EMP = "estimated_employed"
LPR = "estimated_labour_participation_rate"

# Parse dates
for df in (df_india, df_2020):
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%d-%m-%Y", errors="coerce")

# Drop fully-NA rows (28 blank rows in df_india)
df_india.dropna(subset=["region", "date"], inplace=True)

# Duplicates
df_india.drop_duplicates(inplace=True)
df_2020.drop_duplicates(inplace=True)

# Fill Area NaN
if "area" in df_india.columns:
    df_india["area"] = df_india["area"].fillna("Overall")

# Rename zone column in df_2020
df_2020.rename(columns={"region_1": "zone"}, inplace=True)

# Temporal helpers
for df in (df_india, df_2020):
    df["year"]     = df["date"].dt.year
    df["month"]    = df["date"].dt.month
    df["month_nm"] = df["date"].dt.strftime("%b")
    df["quarter"]  = df["date"].dt.quarter

# COVID period flag
covid_start = pd.Timestamp("2020-03-01")
covid_end   = pd.Timestamp("2020-06-30")
def period_flag(d):
    if d >= covid_start and d <= covid_end:
        return "COVID (Mar–Jun 2020)"
    elif d > covid_end:
        return "Recovery (Jul–Nov 2020)"
    return "Pre-COVID"
df_2020["period"] = df_2020["date"].apply(period_flag)

period_order = ["Pre-COVID", "COVID (Mar–Jun 2020)", "Recovery (Jul–Nov 2020)"]

print(f"  df_india : {len(df_india)} rows | {df_india['date'].min().date()} → {df_india['date'].max().date()}")
print(f"  df_2020  : {len(df_2020)} rows  | {df_2020['date'].min().date()} → {df_2020['date'].max().date()}")
print(f"  States in df_india : {df_india['region'].nunique()}")
print(f"  States in df_2020  : {df_2020['region'].nunique()}")
print(f"  Missing in df_india: {df_india[[UR,EMP,LPR]].isnull().sum().to_dict()}")
print(f"  Missing in df_2020 : {df_2020[[UR,EMP,LPR]].isnull().sum().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/9] Dataset Overview")
print("\n  ── df_india descriptive stats ──")
print(df_india[[UR, EMP, LPR]].describe().round(2).to_string())
print("\n  ── df_2020 descriptive stats ──")
print(df_2020[[UR, EMP, LPR]].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. DISTRIBUTION & OUTLIER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/9] Distribution & Outlier Analysis…")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Distribution & Outlier Analysis — Unemployment Rate (%)",
             fontsize=16, fontweight="bold", color=DARK, y=1.01)

# Histogram + KDE – df_india
ax = axes[0, 0]
data_all = df_india[UR].dropna()
ax.hist(data_all, bins=30, color=ACCENT, edgecolor="white", alpha=0.8,
        density=True, label="Histogram")
data_all.plot.kde(ax=ax, color=RED, linewidth=2.2, label="KDE")
ax.set_title("Distribution: All India Data")
ax.set_xlabel("Unemployment Rate (%)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.4)

# Histogram + KDE – df_2020
ax = axes[0, 1]
data_2020 = df_2020[UR].dropna()
ax.hist(data_2020, bins=20, color=ORANGE, edgecolor="white", alpha=0.8, density=True)
data_2020.plot.kde(ax=ax, color=DARK, linewidth=2.2)
ax.set_title("Distribution: 2020 Dataset")
ax.set_xlabel("Unemployment Rate (%)")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.4)

# Boxplot by Area
ax = axes[0, 2]
area_list  = ["Rural", "Overall", "Urban"]
area_data  = [df_india[df_india["area"] == a][UR].dropna().tolist() for a in area_list]
area_cols  = [GREEN, ACCENT, ORANGE]
bp = ax.boxplot(area_data, labels=area_list, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp["boxes"], area_cols):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.set_title("Unemployment Rate by Area Type")
ax.set_ylabel("Unemployment Rate (%)")
ax.grid(True, axis="y", alpha=0.4)

# Boxplot by COVID Period
ax = axes[1, 0]
p_colors = [GREEN, RED, ORANGE]
p_data = [df_2020[df_2020["period"] == p][UR].dropna().tolist() for p in period_order]
bp2 = ax.boxplot(p_data,
                 labels=["Pre-COVID\n(Jan–Feb)", "COVID\n(Mar–Jun)", "Recovery\n(Jul–Nov)"],
                 patch_artist=True,
                 medianprops=dict(color="white", linewidth=2))
for patch, c in zip(bp2["boxes"], p_colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.set_title("Unemployment by COVID Phase")
ax.set_ylabel("Unemployment Rate (%)")
ax.grid(True, axis="y", alpha=0.4)

# IQR Outlier count
ax = axes[1, 1]
Q1, Q3 = data_all.quantile(0.25), data_all.quantile(0.75)
IQR = Q3 - Q1
outliers     = data_all[(data_all < Q1 - 1.5*IQR) | (data_all > Q3 + 1.5*IQR)]
non_outliers = data_all[(data_all >= Q1 - 1.5*IQR) & (data_all <= Q3 + 1.5*IQR)]
bars_iqr = ax.bar(["Normal\nValues", "Outliers\n(IQR)"],
                  [len(non_outliers), len(outliers)],
                  color=[ACCENT, RED], alpha=0.85, edgecolor="white", width=0.5)
for bar, val in zip(bars_iqr, [len(non_outliers), len(outliers)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", fontweight="bold", fontsize=12)
ax.set_title(f"Outlier Count (IQR) — {len(outliers)} outliers")
ax.set_ylabel("Count of Records")
ax.grid(True, axis="y", alpha=0.4)

# Labour Participation KDE
ax = axes[1, 2]
df_india[LPR].dropna().plot.kde(ax=ax, color=ACCENT, linewidth=2.2, label="All India")
df_2020[LPR].dropna().plot.kde(ax=ax, color=ORANGE, linewidth=2.2, label="2020 Dataset")
ax.set_title("Labour Participation Rate Distribution")
ax.set_xlabel("Labour Participation Rate (%)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.4)

plt.tight_layout()
save(fig, "01_distribution_outlier.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NATIONAL UNEMPLOYMENT TREND & COVID
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/9] National Trend & COVID Analysis…")

monthly_nat = (df_2020.groupby("date")[UR].mean()
                       .reset_index().sort_values("date"))
monthly_nat.columns = ["date", "unemp_rate"]
monthly_nat["rolling_3m"] = monthly_nat["unemp_rate"].rolling(3, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 11))
fig.suptitle("National Unemployment Trend & COVID-19 Impact — India 2020",
             fontsize=15, fontweight="bold", color=DARK)

# Line chart with COVID shading
ax = axes[0]
ax.fill_between(monthly_nat["date"], monthly_nat["unemp_rate"],
                color=ACCENT, alpha=0.12)
ax.plot(monthly_nat["date"], monthly_nat["unemp_rate"],
        color=ACCENT, linewidth=2.5, marker="o", markersize=7, label="Monthly Avg.")
ax.plot(monthly_nat["date"], monthly_nat["rolling_3m"],
        color=RED, linewidth=2.2, linestyle="--", label="3-Month Rolling Avg.")
ax.axvspan(covid_start, covid_end, color=RED, alpha=0.08, label="COVID Lockdown")
ax.axvline(covid_start, color=RED, linestyle=":", linewidth=1.8)
ax.axvline(covid_end,   color=ORANGE, linestyle=":", linewidth=1.8)

peak_row = monthly_nat.loc[monthly_nat["unemp_rate"].idxmax()]
ax.annotate(f"  PEAK\n  {peak_row['unemp_rate']:.1f}%",
            xy=(peak_row["date"], peak_row["unemp_rate"]),
            xytext=(peak_row["date"] - pd.Timedelta(days=28), peak_row["unemp_rate"] - 7),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.8),
            fontsize=10, color=RED, fontweight="bold")

ax.set_title("Monthly National Unemployment Rate")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend()
ax.grid(True, alpha=0.4)

# Bar chart by month coloured by period
ax = axes[1]
bar_colors = monthly_nat["date"].apply(
    lambda d: RED if covid_start <= d <= covid_end
              else (ORANGE if d > covid_end else GREEN))
bars = ax.bar(monthly_nat["date"].dt.strftime("%b"), monthly_nat["unemp_rate"],
              color=bar_colors, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, monthly_nat["unemp_rate"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
ax.legend(handles=[Patch(facecolor=GREEN, label="Pre-COVID"),
                   Patch(facecolor=RED,   label="COVID Lockdown"),
                   Patch(facecolor=ORANGE, label="Recovery")])
ax.set_title("Monthly Unemployment Rate by COVID Phase (2020)")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_xlabel("Month (2020)")
ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
save(fig, "02_national_trend_covid.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. STATE-WISE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/9] State-wise & Regional Analysis…")

state_avg = (df_2020.groupby("region")[UR].mean()
                    .sort_values(ascending=False)
                    .reset_index())
state_avg.columns = ["state", "avg_unemp"]
nat_mean = state_avg["avg_unemp"].mean()
nat_std  = state_avg["avg_unemp"].std()

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle("State-wise Unemployment Rate — India 2020",
             fontsize=15, fontweight="bold", color=DARK)

ax = axes[0]
s_colors = [RED if v > nat_mean + nat_std
            else (ORANGE if v > nat_mean else GREEN)
            for v in state_avg["avg_unemp"]]
ax.barh(state_avg["state"], state_avg["avg_unemp"],
        color=s_colors, edgecolor="white", alpha=0.9)
ax.axvline(nat_mean, color=DARK, linestyle="--", linewidth=1.8,
           label=f"Nat. Avg: {nat_mean:.1f}%")
for i, (_, row) in enumerate(state_avg.iterrows()):
    ax.text(row["avg_unemp"] + 0.2, i, f"{row['avg_unemp']:.1f}%",
            va="center", fontsize=7.5, fontweight="bold")
ax.set_title("Average Unemployment Rate by State (2020)")
ax.set_xlabel("Average Unemployment Rate (%)")
ax.legend()
ax.grid(True, axis="x", alpha=0.4)
ax.invert_yaxis()

ax = axes[1]
top10    = state_avg.head(10)
bottom10 = state_avg.tail(10)
combined = pd.concat([top10.assign(grp="Highest 10"), bottom10.assign(grp="Lowest 10")])
ax.barh(combined["state"], combined["avg_unemp"],
        color=combined["grp"].map({"Highest 10": RED, "Lowest 10": GREEN}),
        edgecolor="white", alpha=0.9)
ax.legend(handles=[Patch(facecolor=RED, label="Highest 10"),
                   Patch(facecolor=GREEN, label="Lowest 10")])
ax.set_title("Top 10 Most & Least Affected States")
ax.set_xlabel("Average Unemployment Rate (%)")
ax.grid(True, axis="x", alpha=0.4)

plt.tight_layout()
save(fig, "03_statewise_analysis.png")

# Zone analysis
zone_colors = {"North":"#2563EB","South":"#16A34A","East":"#D97706",
               "West":"#7C3AED","Northeast":"#DC2626"}
zone_monthly = (df_2020.groupby(["zone","date"])[UR]
                        .mean().reset_index().sort_values("date"))
zone_monthly.columns = ["zone","date","unemp_rate"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Zone-wise Unemployment Trends — India 2020",
             fontsize=14, fontweight="bold", color=DARK)

ax = axes[0]
for zone, grp in zone_monthly.groupby("zone"):
    ax.plot(grp["date"], grp["unemp_rate"], marker="o", markersize=4,
            linewidth=2.2, label=zone, color=zone_colors.get(zone, ACCENT))
ax.axvspan(covid_start, covid_end, color=RED, alpha=0.08)
ax.set_title("Monthly Unemployment Rate by Zone")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend()
ax.grid(True, alpha=0.4)

ax = axes[1]
zone_phase = (df_2020.groupby(["zone","period"])[UR]
                     .mean().unstack("period")
                     .reindex(columns=period_order))
zone_phase.plot(kind="bar", ax=ax,
                color=[GREEN, RED, ORANGE], edgecolor="white", alpha=0.9)
ax.set_title("Zone Average Unemployment by COVID Phase")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_xlabel("Zone")
ax.tick_params(axis="x", rotation=0)
ax.legend(["Pre-COVID","COVID Peak","Recovery"], loc="upper left")
ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
save(fig, "04_zone_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. COVID IMPACT DEEP-DIVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/9] COVID-19 Impact Deep-Dive…")

pre_avg   = df_2020[df_2020["period"] == "Pre-COVID"][UR].mean()
covid_avg = df_2020[df_2020["period"] == "COVID (Mar–Jun 2020)"][UR].mean()
rec_avg   = df_2020[df_2020["period"] == "Recovery (Jul–Nov 2020)"][UR].mean()
spike_pct = ((covid_avg - pre_avg) / pre_avg) * 100
rec_pct   = ((covid_avg - rec_avg) / covid_avg) * 100

print(f"  Pre-COVID avg     : {pre_avg:.2f}%")
print(f"  COVID peak avg    : {covid_avg:.2f}%")
print(f"  Recovery avg      : {rec_avg:.2f}%")
print(f"  Spike Pre→COVID   : +{spike_pct:.1f}%")
print(f"  Recovery from peak: -{rec_pct:.1f}%")

# State spike analysis
state_pre   = df_2020[df_2020["period"]=="Pre-COVID"].groupby("region")[UR].mean()
state_covid = df_2020[df_2020["period"]=="COVID (Mar–Jun 2020)"].groupby("region")[UR].mean()
common_st   = state_pre.index.intersection(state_covid.index)
spike_state = ((state_covid[common_st] - state_pre[common_st]) / state_pre[common_st] * 100)
spike_state = spike_state.sort_values(ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("COVID-19 Impact on Unemployment — Quantified Analysis",
             fontsize=15, fontweight="bold", color=DARK)

# Phase comparison
ax = axes[0, 0]
bar_vals  = [pre_avg, covid_avg, rec_avg]
bar_lbls  = ["Pre-COVID\n(Jan–Feb)", "COVID Peak\n(Mar–Jun)", "Recovery\n(Jul–Nov)"]
bar_clrs  = [GREEN, RED, ORANGE]
bars_ph   = ax.bar(bar_lbls, bar_vals, color=bar_clrs, edgecolor="white",
                   alpha=0.9, width=0.5)
for bar, val in zip(bars_ph, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_title("Average Unemployment by COVID Phase (2020)")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_ylim(0, covid_avg * 1.3)
ax.grid(True, axis="y", alpha=0.4)

# State-level spike %
ax = axes[0, 1]
spike_colors = [RED if v > 0 else GREEN for v in spike_state.values]
ax.barh(spike_state.index, spike_state.values,
        color=spike_colors, edgecolor="white", alpha=0.9)
ax.axvline(0, color=DARK, linewidth=1.2)
ax.set_title("% Change in Unemployment: Pre → COVID Peak, by State")
ax.set_xlabel("% Change in Unemployment Rate")
ax.grid(True, axis="x", alpha=0.4)

# Rural vs Urban trend (df_india)
ax = axes[1, 0]
area_monthly = (df_india[df_india["area"].isin(["Rural","Urban"])]
                .groupby(["date","area"])[UR]
                .mean().reset_index().sort_values("date"))
for area, grp in area_monthly.groupby("area"):
    col = GREEN if area == "Rural" else ACCENT
    ax.plot(grp["date"], grp[UR], marker="o", markersize=3.5,
            linewidth=2, label=area, color=col)
ax.axvspan(covid_start, covid_end, color=RED, alpha=0.08, label="Lockdown")
ax.set_title("Rural vs Urban Unemployment Trend")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.4)

# Heatmap: State × Month
ax = axes[1, 1]
pivot = (df_2020.pivot_table(index="region", columns="month",
                              values=UR, aggfunc="mean").round(1))
month_nm_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov"}
pivot.columns = [month_nm_map.get(c, str(c)) for c in pivot.columns]
sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.3, annot_kws={"size": 6.5},
            cbar_kws={"label": "Unemp. Rate (%)"})
ax.set_title("State × Month Unemployment Heatmap (2020)")
ax.set_xlabel("Month")
ax.set_ylabel("State")
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
save(fig, "05_covid_impact.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TIME-SERIES & TREND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/9] Time-Series & Trend Analysis…")

# Build full combined time series
nat_india = (df_india[df_india["area"]=="Overall"]
             .groupby("date")[UR].mean().reset_index()
             .sort_values("date"))
nat_india.columns = ["date","unemp_rate"]
nat_2020  = monthly_nat[["date","unemp_rate"]].copy()

combined_ts = (pd.concat([nat_india, nat_2020])
               .drop_duplicates("date")
               .sort_values("date")
               .reset_index(drop=True))
combined_ts["rolling_3"] = combined_ts["unemp_rate"].rolling(3, min_periods=1).mean()
combined_ts["rolling_6"] = combined_ts["unemp_rate"].rolling(6, min_periods=1).mean()
combined_ts["mom_change"]= combined_ts["unemp_rate"].pct_change(1) * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Time-Series & Trend Analysis — India Unemployment",
             fontsize=15, fontweight="bold", color=DARK)

# Full trend with moving averages
ax = axes[0, 0]
ax.fill_between(combined_ts["date"], combined_ts["unemp_rate"],
                color=ACCENT, alpha=0.10)
ax.plot(combined_ts["date"], combined_ts["unemp_rate"],
        color=ACCENT, linewidth=1.8, marker="o", markersize=4,
        alpha=0.75, label="Monthly")
ax.plot(combined_ts["date"], combined_ts["rolling_3"],
        color=RED, linewidth=2.2, linestyle="--", label="3-Month MA")
ax.plot(combined_ts["date"], combined_ts["rolling_6"],
        color=ORANGE, linewidth=2.2, linestyle="-.", label="6-Month MA")
ax.axvspan(covid_start, covid_end, color=RED, alpha=0.08, label="COVID Lockdown")
ax.set_title("Full Time-Series with Moving Averages (2019–2020)")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# Quarterly averages
ax = axes[0, 1]
df_india["q_label"] = df_india["date"].dt.to_period("Q").astype(str)
q_india = (df_india[df_india["area"]=="Overall"]
           .groupby("q_label")[UR].mean().reset_index())
q_india.columns = ["q_label","unemp_rate"]
q_2020  = (df_2020.assign(q_label=df_2020["date"].dt.to_period("Q").astype(str))
           .groupby("q_label")[UR].mean().reset_index())
q_2020.columns = ["q_label","unemp_rate"]
q_all = (pd.concat([q_india, q_2020]).drop_duplicates("q_label")
          .sort_values("q_label"))
q_colors = [RED if "2020" in q else ACCENT for q in q_all["q_label"]]
ax.bar(q_all["q_label"], q_all["unemp_rate"], color=q_colors,
       edgecolor="white", alpha=0.9)
ax.set_title("Quarterly Average Unemployment")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_xlabel("Quarter")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, axis="y", alpha=0.4)
ax.legend(handles=[Patch(facecolor=ACCENT, label="2019"),
                   Patch(facecolor=RED,   label="2020")])

# Month-over-Month change
ax = axes[1, 0]
mom_valid = combined_ts["mom_change"].dropna()
mom_dates = combined_ts.loc[mom_valid.index, "date"].dt.strftime("%b %y")
m_colors  = [RED if v > 0 else GREEN for v in mom_valid]
ax.bar(mom_dates, mom_valid, color=m_colors, edgecolor="white", alpha=0.9)
ax.axhline(0, color=DARK, linewidth=1.2)
ax.set_title("Month-over-Month Change in Unemployment (%)")
ax.set_ylabel("MoM Change (%)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, axis="y", alpha=0.4)

# Seasonal pattern
ax = axes[1, 1]
seasonal = (df_india[df_india["area"]=="Overall"]
            .groupby("month")[UR].mean())
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
ax.plot(seasonal.index, seasonal.values, color=ACCENT,
        marker="o", markersize=8, linewidth=2.5)
ax.fill_between(seasonal.index, seasonal.values, alpha=0.12, color=ACCENT)
ax.set_xticks(seasonal.index)
ax.set_xticklabels([month_names[m-1] for m in seasonal.index])
ax.set_title("Seasonal Pattern: Avg Unemployment by Month")
ax.set_ylabel("Avg Unemployment Rate (%)")
ax.set_xlabel("Month")
ax.grid(True, alpha=0.4)

plt.tight_layout()
save(fig, "06_trend_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CORRELATION & ADVANCED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/9] Correlation & Advanced Analysis…")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Correlation, Labour Market & Advanced Insights",
             fontsize=15, fontweight="bold", color=DARK)

# Correlation heatmap
ax = axes[0, 0]
corr_cols = df_2020[[UR, EMP, LPR]].dropna().corr()
sns.heatmap(corr_cols, ax=ax, annot=True, fmt=".3f",
            cmap="coolwarm", center=0, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 12, "weight": "bold"})
ax.set_title("Correlation Matrix — Key Variables (2020)")
ax.tick_params(axis="both", labelsize=8)

# Scatter: Unemployment vs Labour Participation
ax = axes[0, 1]
for zone, grp in df_2020.dropna(subset=[LPR, UR]).groupby("zone"):
    ax.scatter(grp[LPR], grp[UR],
               label=zone, alpha=0.6, s=35,
               color=zone_colors.get(zone, ACCENT))
x_all = df_2020[LPR].dropna()
y_all = df_2020.loc[x_all.index, UR].dropna()
idx   = x_all.index.intersection(y_all.index)
coeffs = np.polyfit(x_all[idx], y_all[idx], 1)
xfit   = np.linspace(x_all.min(), x_all.max(), 100)
ax.plot(xfit, np.polyval(coeffs, xfit), color=DARK,
        linewidth=2, linestyle="--", label="Trend")
ax.set_title("Unemployment vs Labour Participation Rate")
ax.set_xlabel("Labour Participation Rate (%)")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.4)

# Rural vs Urban grouped bars
ax = axes[1, 0]
ru_data = (df_india[df_india["area"].isin(["Rural","Urban"])]
           .groupby(["area","month"])[UR].mean().unstack("area"))
months_avail = ru_data.index.tolist()
x_pos = np.arange(len(months_avail))
w     = 0.38
if "Rural" in ru_data.columns:
    ax.bar(x_pos - w/2, ru_data["Rural"], width=w, label="Rural",
           color=GREEN, alpha=0.85, edgecolor="white")
if "Urban" in ru_data.columns:
    ax.bar(x_pos + w/2, ru_data["Urban"], width=w, label="Urban",
           color=ACCENT, alpha=0.85, edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels([month_names[m-1] for m in months_avail], rotation=45)
ax.set_title("Rural vs Urban Monthly Unemployment (2019–2020)")
ax.set_ylabel("Avg Unemployment Rate (%)")
ax.legend()
ax.grid(True, axis="y", alpha=0.4)

# Top 5 most-spiked states trend
ax = axes[1, 1]
top5 = spike_state.head(5).index.tolist()
for st in top5:
    grp = df_2020[df_2020["region"]==st].groupby("date")[UR].mean()
    ax.plot(grp.index, grp.values, marker="o", markersize=4,
            linewidth=1.8, label=st)
ax.axvspan(covid_start, covid_end, color=RED, alpha=0.08)
ax.set_title("Top 5 COVID-Affected States — Monthly Trend (2020)")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

plt.tight_layout()
save(fig, "07_correlation_advanced.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXECUTIVE SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/9] Generating Executive Summary Dashboard…")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(DARK)

kpis = [
    ("Avg Pre-COVID",    f"{pre_avg:.1f}%",           GREEN),
    ("COVID Peak Avg",   f"{covid_avg:.1f}%",          RED),
    ("Recovery Avg",     f"{rec_avg:.1f}%",            ORANGE),
    ("Unemployment Spike",f"+{spike_pct:.0f}%",        RED),
    ("Peak Recovery",    f"-{rec_pct:.0f}%",           GREEN),
    ("States Analysed",  f"{df_2020['region'].nunique()}", ACCENT),
]
for i, (label, value, color) in enumerate(kpis):
    a = fig.add_axes([0.01 + i*0.165, 0.83, 0.15, 0.12])
    a.set_facecolor(color)
    a.text(0.5, 0.65, value, ha="center", va="center",
           fontsize=22, fontweight="bold", color="white", transform=a.transAxes)
    a.text(0.5, 0.20, label, ha="center", va="center",
           fontsize=8.5, color="white", transform=a.transAxes)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)

# Main trend
ax1 = fig.add_axes([0.01, 0.45, 0.45, 0.35])
ax1.set_facecolor("#1E293B")
ax1.fill_between(monthly_nat["date"], monthly_nat["unemp_rate"], color=ACCENT, alpha=0.2)
ax1.plot(monthly_nat["date"], monthly_nat["unemp_rate"], color=ACCENT,
         linewidth=2.5, marker="o", markersize=5)
ax1.plot(monthly_nat["date"], monthly_nat["rolling_3m"], color=RED,
         linewidth=2, linestyle="--")
ax1.axvspan(covid_start, covid_end, color=RED, alpha=0.15)
ax1.set_title("National Unemployment Trend 2020", color="white", fontsize=11)
ax1.tick_params(colors="white")
ax1.grid(True, alpha=0.2, color="white")
for sp in ax1.spines.values(): sp.set_color("#4B5563")

# Zone trend
ax2 = fig.add_axes([0.52, 0.45, 0.46, 0.35])
ax2.set_facecolor("#1E293B")
for zone, grp in zone_monthly.groupby("zone"):
    ax2.plot(grp["date"], grp["unemp_rate"], linewidth=2.2,
             label=zone, color=zone_colors.get(zone, ACCENT))
ax2.axvspan(covid_start, covid_end, color=RED, alpha=0.12)
ax2.set_title("Zone-wise Unemployment Trend", color="white", fontsize=11)
ax2.tick_params(colors="white")
leg2 = ax2.legend(fontsize=8, facecolor="#374151", edgecolor="none")
for t in leg2.get_texts(): t.set_color("white")
ax2.grid(True, alpha=0.2, color="white")
for sp in ax2.spines.values(): sp.set_color("#4B5563")

# Top-10 states
ax3 = fig.add_axes([0.01, 0.04, 0.45, 0.35])
ax3.set_facecolor("#1E293B")
t10 = state_avg.head(10)
ax3.barh(t10["state"], t10["avg_unemp"],
         color=[RED if v > nat_mean else ORANGE for v in t10["avg_unemp"]],
         edgecolor="none", alpha=0.9)
ax3.invert_yaxis()
ax3.set_title("Top 10 States by Unemployment (2020)", color="white", fontsize=11)
ax3.tick_params(colors="white"); ax3.xaxis.label.set_color("white")
for sp in ax3.spines.values(): sp.set_color("#4B5563")
ax3.grid(True, axis="x", alpha=0.2, color="white")

# Heatmap (top 15 states)
ax4 = fig.add_axes([0.52, 0.04, 0.46, 0.35])
ax4.set_facecolor("#1E293B")
pivot_mini = pivot.head(15)
sns.heatmap(pivot_mini, ax=ax4, cmap="YlOrRd", linewidths=0.2,
            annot=True, fmt=".0f", annot_kws={"size": 6},
            cbar_kws={"shrink": 0.7})
ax4.set_title("State × Month Heatmap (Top 15)", color="white", fontsize=11)
ax4.tick_params(colors="white", labelsize=7)

fig.text(0.5, 0.975, "UNEMPLOYMENT ANALYSIS DASHBOARD — INDIA 2019–2020",
         ha="center", va="center", fontsize=16, fontweight="bold", color="white")

save(fig, "08_executive_dashboard.png")

print("\n" + "=" * 60)
print("  ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("=" * 60)

print("\n  KEY QUANTIFIED INSIGHTS")
print(f"  • Pre-COVID avg unemployment   : {pre_avg:.2f}%")
print(f"  • COVID peak avg unemployment  : {covid_avg:.2f}%")
print(f"  • Recovery avg unemployment    : {rec_avg:.2f}%")
print(f"  • Absolute spike (Pre→Peak)    : +{covid_avg-pre_avg:.2f} pp")
print(f"  • Percentage spike             : +{spike_pct:.1f}%")
print(f"  • Recovery from peak           : -{rec_pct:.1f}%")
print(f"  • Highest-unemp state (2020)   : {state_avg.iloc[0]['state']} ({state_avg.iloc[0]['avg_unemp']:.1f}%)")
print(f"  • Lowest-unemp state (2020)    : {state_avg.iloc[-1]['state']} ({state_avg.iloc[-1]['avg_unemp']:.1f}%)")
print(f"  • Biggest spike state          : {spike_state.index[0]} (+{spike_state.iloc[0]:.0f}%)")
