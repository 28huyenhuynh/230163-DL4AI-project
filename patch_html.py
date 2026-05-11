"""Replace t53 XSL append with a clean standalone HTML generator."""
import json, pathlib

NB = pathlib.Path(r"c:\Users\Admin\230163-DL4AI-project\230163_project_notebook.ipynb")
nb = json.loads(NB.read_bytes().decode("utf-8"))

# Full replacement of t53 source
T53_SRC = r'''import xml.etree.ElementTree as ET
from xml.dom import minidom

# ── Build pipeline.xml (machine-readable) ────────────────────────────────
def _sub(parent, tag, text=None, **attrs):
    el = ET.SubElement(parent, tag, **attrs)
    if text: el.text = text
    return el

root = ET.Element("pipeline", name="stock-prediction-pipeline", version="1.0",
                  xmlns="urn:stock-prediction:workflow:1.0")
meta = _sub(root, "metadata")
_sub(meta, "author",      "Huynh Nhat Huyen")
_sub(meta, "student_id",  "230163")
_sub(meta, "course",      "CS313 Deep Learning for AI")
_sub(meta, "description", "Automated ML pipeline for Vietnam and Nasdaq stock price prediction")
_sub(meta, "created",     "2026-05-11")

tools_el = _sub(root, "tools")
tool_defs = [
    ("airbyte",    "ingestion",          "0.50", "Open-source EL platform. Extracts daily OHLCV data from exchange APIs and loads into PostgreSQL. Handles schema evolution, retries, and incremental syncs with no custom scraper code."),
    ("postgresql", "relational_storage", "16",   "Relational database for structured time-series data. Stores raw OHLCV, cleaned features, and daily predictions in typed, indexed tables. ACID transactions prevent partial writes."),
    ("dbt",        "transformation",     "1.8",  "SQL-first transformation framework. Builds version-controlled feature tables (SMA, RSI, MACD, BB) from raw PostgreSQL tables. Equivalent to clean_ohlcv() + add_indicators() inside the database."),
    ("mongodb",    "unstructured_storage","7",   "Document-oriented NoSQL database. Stores raw API JSON responses, news, model experiment logs (MAE, RMSE, Dir accuracy), and model registry metadata."),
    ("airflow",    "orchestration",      "2.9",  "Workflow scheduler that runs pipeline steps as DAGs on a cron schedule. Manages task dependencies, retries, SLA monitoring, and failure alerting."),
]
for tid, role, ver, desc in tool_defs:
    t = _sub(tools_el, "tool", id=tid, role=role, version=ver)
    _sub(t, "description", desc)

stages_el = _sub(root, "stages")
stage_defs = [
    ("1","ingestion",     "airbyte",          "Extract OHLCV data from exchange APIs into raw.stock_prices (PostgreSQL).",       [("source","Yahoo Finance / VNDirect API"),("destination","PostgreSQL: raw.stock_prices"),("schedule","Daily 17:30")]),
    ("2","transformation","dbt",              "Clean raw prices and compute technical indicators inside the database.",           [("input","raw.stock_prices"),("output","features.technical, features.ml_ready"),("schedule","Daily 17:45")]),
    ("3","training",      "python_lstm",      "Retrain LSTM models on the latest data window; log metrics to MongoDB.",          [("input","features.ml_ready"),("output","saved_models/"),("logging","MongoDB: experiments.runs"),("schedule","Weekly Sun 02:00")]),
    ("4","inference",     "python_lstm",      "Generate next-day price predictions using the latest saved model.",               [("input","features.ml_ready (last WINDOW rows)"),("output","predictions.daily_forecast"),("schedule","Daily 18:00")]),
    ("5","serving",       "fastapi_streamlit","Expose predictions via REST API and interactive Streamlit dashboard.",            [("fastapi","GET /health  POST /predict/price"),("streamlit","Candlestick chart + prediction button"),("schedule","Always-on")]),
    ("6","monitoring",    "airflow",          "Track prediction quality daily and detect model drift.",                          [("metric","Rolling-7d MAE and direction accuracy"),("alert","Email/Slack if MAE > 2x baseline"),("logging","MongoDB: monitoring.daily_metrics")]),
]
for sid, name, tool, desc, params in stage_defs:
    s = _sub(stages_el, "stage", id=sid, name=name, tool=tool)
    _sub(s, "description", desc)
    p = _sub(s, "parameters")
    for k, v in params:
        _sub(p, "param", v, name=k)

xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
xml_out = "\n".join(l for l in xml_str.splitlines() if l.strip())
with open("pipeline.xml", "w", encoding="utf-8") as f:
    f.write(xml_out)
print("pipeline.xml written.")

# ── Build pipeline.html (human-readable) ─────────────────────────────────
refs = [
    ("1","Apache Airflow Project","Apache Airflow Documentation","2024","https://airflow.apache.org/docs/"),
    ("2","Airbyte, Inc.","Airbyte Documentation","2024","https://docs.airbyte.com/"),
    ("3","dbt Labs","dbt Documentation","2024","https://docs.getdbt.com/"),
    ("4","MongoDB, Inc.","MongoDB Documentation","2024","https://www.mongodb.com/docs/"),
    ("5","The PostgreSQL Global Development Group","PostgreSQL 16 Documentation","2024","https://www.postgresql.org/docs/16/"),
    ("6","Sculley, D. et al.","Hidden Technical Debt in Machine Learning Systems. NeurIPS 28","2015","https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html"),
    ("7","Zaharia, M. et al.","Accelerating the ML Lifecycle with MLflow. IEEE Data Eng. Bulletin 41(4)","2018","https://databricks.com/wp-content/uploads/2018/12/MLflow-Dec-2018.pdf"),
]

dag_daily  = [("sync_ohlcv","AirbyteTriggerSyncOperator",""),("dbt_run","BashOperator","sync_ohlcv"),("lstm_inference","PythonOperator","dbt_run"),("write_predictions","PythonOperator","lstm_inference"),("log_metrics","PythonOperator","write_predictions")]
dag_weekly = [("sync_ohlcv","AirbyteTriggerSyncOperator",""),("dbt_run","BashOperator","sync_ohlcv"),("lstm_retrain","PythonOperator","dbt_run"),("evaluate_model","PythonOperator","lstm_retrain"),("save_artifacts","PythonOperator","evaluate_model")]

tool_colors = {"airbyte":"#e65100","postgresql":"#1565c0","dbt":"#2e7d32","mongodb":"#1b5e20","airflow":"#6a1b9a","python_lstm":"#00695c","fastapi_streamlit":"#37474f"}

def badge(tool):
    c = tool_colors.get(tool, "#455a64")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:10px;font-size:.8em">{tool}</span>'

def dag_html(tasks):
    boxes = "".join(f'''<div style="display:flex;flex-direction:column;align-items:center">
      <div style="background:#1e2a38;border:1px solid #4fc3f7;border-radius:6px;padding:6px 12px;font-size:.85em;min-width:90px;text-align:center">
        <b style="color:#4fc3f7">{t}</b><br/><span style="color:#90a4ae;font-size:.8em">{op.split("Operator")[0]}</span></div>
      {"" if not dep else '<div style="color:#4fc3f7;font-size:1.2em">&#8595;</div>'}
    </div>''' for t, op, dep in tasks)
    return f'<div style="display:flex;flex-direction:column;align-items:flex-start;gap:0">{boxes}</div>'

stage_rows = "".join(f"""<tr>
  <td style="color:#4fc3f7;font-weight:bold;text-align:center">{sid}</td>
  <td><b>{name}</b></td>
  <td>{badge(tool)}</td>
  <td style="color:#cfd8dc">{desc}</td>
  <td style="font-size:.82em;color:#90a4ae">{"<br>".join(f"<b>{k}:</b> {v}" for k,v in params)}</td>
</tr>""" for sid, name, tool, desc, params in stage_defs)

tool_rows = "".join(f"""<tr>
  <td><b style="color:#fff">{tid}</b></td>
  <td>{badge(role)}</td>
  <td style="color:#90a4ae">{ver}</td>
  <td style="color:#cfd8dc">{desc}</td>
</tr>""" for tid, role, ver, desc in tool_defs)

ref_items = "".join(f'<li style="margin:6px 0;color:#90a4ae">[{rid}] {author}. <i>{title}</i>. {year}. <a href="{url}" style="color:#4fc3f7">{url}</a></li>' for rid, author, title, year, url in refs)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>AI Engineering Pipeline — Stock Prediction</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e0e0e0;padding:32px;line-height:1.6}}
  h1{{color:#4fc3f7;font-size:1.8em;border-bottom:2px solid #4fc3f7;padding-bottom:10px;margin-bottom:6px}}
  .subtitle{{color:#90a4ae;margin-bottom:28px;font-size:.95em}}
  h2{{color:#81d4fa;font-size:1.2em;margin:36px 0 14px;border-left:3px solid #4fc3f7;padding-left:10px}}
  table{{width:100%;border-collapse:collapse;margin-bottom:8px}}
  th{{background:#1a2535;color:#4fc3f7;padding:9px 13px;text-align:left;border-bottom:2px solid #2d3f50;font-size:.9em}}
  td{{padding:9px 13px;border-bottom:1px solid #1e2a38;vertical-align:top;font-size:.9em}}
  tr:hover td{{background:#141b26}}
  .flow{{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:14px 0}}
  .flow-box{{background:#1e2a38;border:1px solid #2d3f50;border-radius:6px;padding:8px 14px;font-size:.85em;text-align:center;min-width:110px}}
  .flow-arrow{{color:#4fc3f7;font-size:1.4em;padding:0 2px}}
  .dag-label{{color:#81d4fa;font-weight:bold;font-size:.9em;margin:14px 0 6px}}
  footer{{margin-top:48px;padding-top:16px;border-top:1px solid #1e2a38;color:#546e7a;font-size:.82em}}
</style>
</head>
<body>

<h1>AI Engineering Pipeline &mdash; Stock Price Prediction</h1>
<p class="subtitle">
  <b style="color:#cfd8dc">Student:</b> Huynh Nhat Huyen &nbsp;|&nbsp;
  <b style="color:#cfd8dc">ID:</b> 230163 &nbsp;|&nbsp;
  <b style="color:#cfd8dc">Course:</b> CS313 Deep Learning for AI &nbsp;|&nbsp;
  <b style="color:#cfd8dc">Created:</b> 2026-05-11
</p>

<!-- Pipeline flow diagram -->
<h2>Pipeline Overview</h2>
<div class="flow">
  <div class="flow-box"><b style="color:#e65100">Airbyte</b><br/><span style="color:#90a4ae">Ingestion</span></div>
  <span class="flow-arrow">&#8594;</span>
  <div class="flow-box"><b style="color:#1e88e5">PostgreSQL</b><br/><span style="color:#90a4ae">Raw Storage</span></div>
  <span class="flow-arrow">&#8594;</span>
  <div class="flow-box"><b style="color:#43a047">dbt</b><br/><span style="color:#90a4ae">Transform</span></div>
  <span class="flow-arrow">&#8594;</span>
  <div class="flow-box"><b style="color:#26a69a">LSTM Model</b><br/><span style="color:#90a4ae">Train / Infer</span></div>
  <span class="flow-arrow">&#8594;</span>
  <div class="flow-box"><b style="color:#1e88e5">PostgreSQL</b><br/><span style="color:#90a4ae">Predictions</span></div>
  <span class="flow-arrow">&#8594;</span>
  <div class="flow-box"><b style="color:#546e7a">FastAPI</b><br/><span style="color:#90a4ae">REST API</span></div>
  <span class="flow-arrow">&#8594;</span>
  <div class="flow-box"><b style="color:#546e7a">Streamlit</b><br/><span style="color:#90a4ae">Dashboard</span></div>
</div>
<p style="color:#90a4ae;font-size:.85em;margin-top:4px">
  Unstructured data (API responses, experiment logs, news) &rarr; <b style="color:#43a047">MongoDB</b> &nbsp;|&nbsp;
  All steps orchestrated by <b style="color:#8e24aa">Apache Airflow</b>
</p>

<!-- Tools -->
<h2>Tools</h2>
<table>
  <tr><th>Tool</th><th>Role</th><th>Version</th><th>Description</th></tr>
  {tool_rows}
</table>

<!-- Stages -->
<h2>Pipeline Stages</h2>
<table>
  <tr><th>#</th><th>Stage</th><th>Tool</th><th>Description</th><th>Parameters</th></tr>
  {stage_rows}
</table>

<!-- Airflow DAGs -->
<h2>Airflow DAGs</h2>
<div style="display:flex;gap:48px;flex-wrap:wrap">
  <div>
    <p class="dag-label">daily_pipeline &mdash; <code style="background:#1a2535;padding:2px 6px;border-radius:3px">0 17 * * 1-5</code></p>
    {dag_html(dag_daily)}
  </div>
  <div>
    <p class="dag-label">weekly_retrain &mdash; <code style="background:#1a2535;padding:2px 6px;border-radius:3px">0 2 * * 0</code></p>
    {dag_html(dag_weekly)}
  </div>
</div>

<!-- References -->
<h2>References</h2>
<ol style="padding-left:20px">{ref_items}</ol>

<footer>Generated by 230163_project_notebook.ipynb &mdash; CS313 Deep Learning for AI</footer>
</body></html>"""

with open("pipeline.html", "w", encoding="utf-8") as f:
    f.write(html)
print("pipeline.html written — open in any browser.")
'''

for c in nb["cells"]:
    if c.get("id") == "t53":
        lines = T53_SRC.splitlines(keepends=True)
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1][:-1]
        c["source"] = lines
        print("t53 replaced.")
        break

NB.write_bytes(json.dumps(nb, ensure_ascii=False, indent=1).encode("utf-8"))
print("Patch applied.")
