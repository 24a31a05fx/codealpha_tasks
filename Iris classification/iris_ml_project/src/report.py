"""
report.py
---------
Assembles all visualisations and metrics into a self-contained HTML report.
Designed to be run after the pipeline completes — takes the results dict
produced by models.train_all() and the image dicts from visualiser.
"""

import os
from datetime import datetime

from config import OUTPUT_DIR, FEATURES, FEATURE_DISPLAY, CV_FOLDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tag(val: float, good: float = 0.93) -> str:
    colour = "#43BBAD" if val >= good else "#FF6584"
    return f'<span style="color:{colour};font-weight:700">{val*100:.1f}%</span>'


def _report_rows(report: dict, classes: list) -> str:
    rows = []
    for cls in classes:
        r   = report[cls]
        f1c = "#43BBAD" if r["f1-score"] >= 0.93 else "#FF6584"
        rows.append(
            f"<tr>"
            f"<td><span class='chip'>{cls.replace('Iris-','')}</span></td>"
            f"<td>{r['precision']:.3f}</td>"
            f"<td>{r['recall']:.3f}</td>"
            f"<td style='color:{f1c};font-weight:600'>{r['f1-score']:.3f}</td>"
            f"<td>{int(r['support'])}</td>"
            f"</tr>"
        )
    return "".join(rows)


def _model_cards(results: dict, best_name: str) -> str:
    classes = [k for k in next(iter(results.values()))["report"]
               if k not in ("accuracy", "macro avg", "weighted avg")]
    html = ""
    for name, res in results.items():
        is_best = name == best_name
        html += f"""
        <div class="mcard{'  best-card' if is_best else ''}">
          <div class="mcard-header">
            <span class="mname">{'⭐ ' if is_best else ''}{name}</span>
            {'<span class="badge-best">Best Model</span>' if is_best else ''}
          </div>
          <div class="stat-row">
            <div class="stat-block">
              <span class="sval">{res['acc']*100:.1f}%</span>
              <span class="slabel">Test Accuracy</span>
            </div>
            <div class="stat-block">
              <span class="sval">{res['cv'].mean()*100:.1f}%</span>
              <span class="slabel">CV Mean ({CV_FOLDS}-fold)</span>
            </div>
            <div class="stat-block">
              <span class="sval">±{res['cv'].std()*100:.1f}%</span>
              <span class="slabel">CV Std Dev</span>
            </div>
            <div class="stat-block">
              <span class="sval">{res['report']['weighted avg']['f1-score']:.3f}</span>
              <span class="slabel">Weighted F1</span>
            </div>
          </div>
          <table class="rtable">
            <thead><tr>
              <th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th>
            </tr></thead>
            <tbody>{_report_rows(res['report'], classes)}</tbody>
          </table>
        </div>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def build_report(
    results:      dict,
    best_name:    str,
    summary:      dict,
    images:       dict,
    importances:  dict,
    output_path:  str = None,
) -> str:
    """
    Render and write the HTML report.

    Parameters
    ----------
    results      : from models.train_all()
    best_name    : from models.get_best()
    summary      : from data_loader.get_dataset_summary()
    images       : dict of plot-name -> base64 PNG string
    importances  : from models.get_feature_importances()
    output_path  : override default output location

    Returns
    -------
    path : str  path the file was written to
    """
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "iris_classification_report.html")

    best = results[best_name]
    now  = datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── stats bar ─────────────────────────────────────────────────────────────
    class_counts_html = " ".join(
        f"<span class='chip'>{k.replace('Iris-','')} — {v}</span>"
        for k, v in summary["class_counts"].items()
    )

    # ── describe table ─────────────────────────────────────────────────────────
    desc = summary["describe"]
    stat_keys = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    desc_rows = ""
    for feat in FEATURES:
        cells = "".join(f"<td>{desc[feat][s]:.3f}</td>" for s in stat_keys)
        desc_rows += f"<tr><td><strong>{FEATURE_DISPLAY[feat]}</strong></td>{cells}</tr>"

    # ── importance table ───────────────────────────────────────────────────────
    imp_rows = ""
    for rank, (feat, score) in enumerate(importances.items(), 1):
        bar_w = int(score * 400)
        imp_rows += (
            f"<tr><td>#{rank}</td>"
            f"<td>{FEATURE_DISPLAY.get(feat, feat)}</td>"
            f"<td><div class='imp-bar' style='width:{bar_w}px'></div></td>"
            f"<td>{score:.4f}</td></tr>"
        )

    # ── results summary table ──────────────────────────────────────────────────
    results_rows = ""
    for name, res in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        star = "⭐" if name == best_name else ""
        results_rows += (
            f"<tr>"
            f"<td>{star} {name}</td>"
            f"<td>{_tag(res['acc'])}</td>"
            f"<td>{res['cv'].mean()*100:.1f}%</td>"
            f"<td>±{res['cv'].std()*100:.1f}%</td>"
            f"<td>{res['report']['weighted avg']['f1-score']:.3f}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Iris Flower Classification — ML Report</title>
<style>
/* ── Reset & base ─────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:      #0b1120;
  --surface: #141c2e;
  --card:    #1a2540;
  --border:  #263352;
  --muted:   #7d92b5;
  --text:    #dde5f4;
  --primary: #6C63FF;
  --teal:    #43BBAD;
  --rose:    #FF6584;
  --font:    'Segoe UI', system-ui, -apple-system, sans-serif;
}}
html {{ scroll-behavior: smooth; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.6;
}}

/* ── Layout ──────────────────────────────────────────── */
.wrapper    {{ max-width: 1080px; margin: 0 auto; padding: 40px 24px 80px; }}
.grid-2     {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.grid-4     {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }}
.grid-3     {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }}
@media(max-width:680px) {{
  .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
}}

/* ── Header ──────────────────────────────────────────── */
.page-header {{
  text-align: center;
  padding: 60px 0 50px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 50px;
}}
.page-header h1 {{
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(120deg, #6C63FF 30%, #43BBAD);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 10px;
}}
.page-header .subtitle {{
  color: var(--muted);
  font-size: 1rem;
  margin-bottom: 22px;
}}
.meta-pills {{ display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }}
.meta-pill  {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: .8rem;
  color: var(--muted);
}}

/* ── Section headers ─────────────────────────────────── */
h2.section-title {{
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--primary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 52px 0 18px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}}
h2.section-title::before {{
  content: '';
  display: inline-block;
  width: 4px;
  height: 18px;
  background: var(--primary);
  border-radius: 2px;
}}

/* ── Cards ───────────────────────────────────────────── */
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
}}
.stat-card {{ text-align: center; }}
.stat-card .big-num {{
  font-size: 2.2rem;
  font-weight: 800;
  color: var(--teal);
  line-height: 1.1;
}}
.stat-card .label {{
  font-size: .78rem;
  color: var(--muted);
  margin-top: 5px;
  text-transform: uppercase;
  letter-spacing: .06em;
}}

/* ── Plot images ─────────────────────────────────────── */
.plot-img {{
  width: 100%;
  border-radius: 10px;
  display: block;
  margin-top: 14px;
}}

/* ── Summary results table ───────────────────────────── */
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 9px 12px; text-align: left; }}
th {{
  background: #101828;
  color: var(--muted);
  font-size: .78rem;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 600;
}}
td {{ border-top: 1px solid var(--border); color: #c8d3e8; font-size: .88rem; }}
tr:hover td {{ background: #1f2d45; }}

/* ── Model cards ─────────────────────────────────────── */
.mcard {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
  margin-bottom: 16px;
  transition: border-color .2s;
}}
.mcard:hover {{ border-color: var(--primary); }}
.best-card    {{ border-color: var(--teal); box-shadow: 0 0 18px #43BBAD28; }}
.mcard-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}}
.mname {{ font-weight: 700; color: white; font-size: 1rem; }}
.badge-best {{
  background: #43BBAD22;
  color: var(--teal);
  border: 1px solid #43BBAD55;
  border-radius: 999px;
  padding: 2px 12px;
  font-size: .75rem;
  font-weight: 600;
}}
.stat-row {{
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}
.stat-block   {{ display: flex; flex-direction: column; }}
.sval  {{ font-size: 1.25rem; font-weight: 800; color: var(--primary); }}
.best-card .sval {{ color: var(--teal); }}
.slabel {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
.rtable th {{ font-size: .75rem; padding: 7px 10px; }}
.rtable td {{ font-size: .82rem; padding: 6px 10px; }}

/* ── Chips ───────────────────────────────────────────── */
.chip {{
  display: inline-block;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 2px 8px;
  font-size: .78rem;
  color: #aab8d4;
  margin: 2px;
}}

/* ── Importance bar ──────────────────────────────────── */
.imp-bar {{
  height: 8px;
  background: linear-gradient(90deg, #6C63FF, #43BBAD);
  border-radius: 4px;
  min-width: 6px;
}}

/* ── Footer ──────────────────────────────────────────── */
.footer {{
  text-align: center;
  color: var(--border);
  font-size: .78rem;
  margin-top: 70px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}}
</style>
</head>
<body>
<div class="wrapper">

  <!-- ── HEADER ─────────────────────────────────────────── -->
  <header class="page-header">
    <h1>🌸 Iris Flower Classification</h1>
    <p class="subtitle">Multi-Model ML Pipeline · Comparative Evaluation Report</p>
    <div class="meta-pills">
      <span class="meta-pill">📅 Generated {now}</span>
      <span class="meta-pill">🔬 scikit-learn</span>
      <span class="meta-pill">📊 150 samples · 4 features · 3 classes</span>
      <span class="meta-pill">🏆 Best: {best_name} ({best['acc']*100:.1f}%)</span>
    </div>
  </header>

  <!-- ── DATASET ────────────────────────────────────────── -->
  <h2 class="section-title">Dataset Overview</h2>
  <div class="grid-4">
    <div class="card stat-card">
      <div class="big-num">{summary['n_samples']}</div>
      <div class="label">Total Samples</div>
    </div>
    <div class="card stat-card">
      <div class="big-num">{summary['n_features']}</div>
      <div class="label">Features</div>
    </div>
    <div class="card stat-card">
      <div class="big-num">{summary['n_classes']}</div>
      <div class="label">Target Classes</div>
    </div>
    <div class="card stat-card">
      <div class="big-num">{summary['missing_values']}</div>
      <div class="label">Missing Values</div>
    </div>
  </div>

  <div class="card" style="margin-top:18px">
    <strong style="color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.06em">
      Class Distribution
    </strong>
    <div style="margin-top:10px">{class_counts_html}</div>
  </div>

  <div class="grid-2" style="margin-top:18px">
    <div class="card" style="display:flex;flex-direction:column;align-items:center">
      <img class="plot-img" src="data:image/png;base64,{images['donut']}" alt="class distribution">
    </div>
    <div class="card" style="overflow-x:auto">
      <strong style="color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.06em">
        Descriptive Statistics
      </strong>
      <table style="margin-top:12px">
        <thead><tr>
          <th>Feature</th><th>Mean</th><th>Std</th><th>Min</th>
          <th>25%</th><th>50%</th><th>75%</th><th>Max</th>
        </tr></thead>
        <tbody>{desc_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- ── EDA ────────────────────────────────────────────── -->
  <h2 class="section-title">Exploratory Data Analysis</h2>
  <div class="card">
    <img class="plot-img" src="data:image/png;base64,{images['distributions']}" alt="distributions">
  </div>
  <div class="card" style="margin-top:18px">
    <img class="plot-img" src="data:image/png;base64,{images['boxplots']}" alt="boxplots">
  </div>
  <div class="card" style="margin-top:18px">
    <img class="plot-img" src="data:image/png;base64,{images['pairplot']}" alt="pairplot">
  </div>
  <div class="card" style="margin-top:18px">
    <img class="plot-img" src="data:image/png;base64,{images['correlation']}" alt="correlation">
  </div>

  <!-- ── RESULTS SUMMARY ────────────────────────────────── -->
  <h2 class="section-title">Model Comparison Summary</h2>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Model</th><th>Test Acc</th><th>CV Mean</th><th>CV Std</th><th>Weighted F1</th>
      </tr></thead>
      <tbody>{results_rows}</tbody>
    </table>
  </div>

  <div class="card" style="margin-top:18px">
    <img class="plot-img" src="data:image/png;base64,{images['comparison']}" alt="comparison">
  </div>
  <div class="card" style="margin-top:18px">
    <img class="plot-img" src="data:image/png;base64,{images['cv']}" alt="cross validation">
  </div>

  <!-- ── BEST MODEL ─────────────────────────────────────── -->
  <h2 class="section-title">Best Model Deep-Dive — {best_name}</h2>
  <div class="grid-2">
    <div class="card">
      <img class="plot-img" src="data:image/png;base64,{images['confusion']}" alt="confusion matrix">
    </div>
    <div class="card">
      <img class="plot-img" src="data:image/png;base64,{images['importance']}" alt="feature importance">
      <div style="margin-top:18px;overflow-x:auto">
        <table>
          <thead><tr><th>#</th><th>Feature</th><th>Bar</th><th>Score</th></tr></thead>
          <tbody>{imp_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── ALL MODEL CARDS ────────────────────────────────── -->
  <h2 class="section-title">Detailed Per-Model Metrics</h2>
  {_model_cards(results, best_name)}

  <!-- ── FOOTER ─────────────────────────────────────────── -->
  <div class="footer">
    Iris Flower Classification Report &nbsp;·&nbsp;
    scikit-learn {'{sklearn.__version__}' if False else ''}&nbsp;·&nbsp;
    UCI Iris Dataset &nbsp;·&nbsp; {now}
  </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return output_path
