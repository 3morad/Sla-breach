from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

W, H = A4
doc = SimpleDocTemplate(
    "NetPulse_Dashboard_Overview.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

# ── colours ──────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0d1b2a")
TEAL   = colors.HexColor("#10b981")
BLUE   = colors.HexColor("#3b82f6")
RED    = colors.HexColor("#ef4444")
AMBER  = colors.HexColor("#f59e0b")
PURPLE = colors.HexColor("#8b5cf6")
SLATE  = colors.HexColor("#64748b")
WHITE  = colors.white
LIGHT  = colors.HexColor("#f1f5f9")

styles = getSampleStyleSheet()

def S(name, **kw):
    base = styles[name] if name in styles else styles["Normal"]
    return ParagraphStyle(name + str(id(kw)), parent=base, **kw)

title_s    = S("Normal", fontSize=26, textColor=TEAL,  leading=32, alignment=TA_CENTER, fontName="Helvetica-Bold")
sub_s      = S("Normal", fontSize=12, textColor=SLATE, leading=16, alignment=TA_CENTER)
h1_s       = S("Normal", fontSize=14, textColor=TEAL,  leading=20, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)
h2_s       = S("Normal", fontSize=11, textColor=BLUE,  leading=16, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=2)
body_s     = S("Normal", fontSize=9,  textColor=colors.HexColor("#1e293b"), leading=14)
code_s     = S("Normal", fontSize=8,  textColor=colors.HexColor("#334155"), leading=13, fontName="Courier", leftIndent=12)
label_s    = S("Normal", fontSize=7,  textColor=SLATE, leading=10, fontName="Helvetica-Bold")
link_s     = S("Normal", fontSize=10, textColor=BLUE,  leading=14, alignment=TA_CENTER)

def HR(color=TEAL, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=6)

def heading1(txt): return Paragraph(txt, h1_s)
def heading2(txt): return Paragraph(txt, h2_s)
def body(txt):     return Paragraph(txt, body_s)
def code(txt):     return Paragraph(txt, code_s)
def sp(n=6):       return Spacer(1, n)

# ── table helper ──────────────────────────────────────────────────────────────
def styled_table(data, col_widths, header_color=NAVY):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",  (0,0), (-1,0),  header_color),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  8),
        ("FONTSIZE",    (0,1), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(style))
    return t

# ─────────────────────────────────────────────────────────────────────────────
story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story += [
    sp(30),
    Paragraph("🛜 NetPulse Analytics", title_s),
    sp(8),
    Paragraph("Telecom Network Intelligence Dashboard", sub_s),
    sp(4),
    Paragraph("Project Overview &amp; Teammate Contribution Guide", sub_s),
    sp(20),
    HR(TEAL, 1.5),
    sp(6),
    Paragraph("GitHub Repository", label_s),
    sp(2),
    Paragraph("https://github.com/3morad/Sla-breach", link_s),
    sp(4),
    Paragraph("Branch: <b>claude/xgboost-sla-streamlit-app-h10XP</b>", S("Normal", fontSize=9, textColor=SLATE, alignment=TA_CENTER)),
    sp(4),
    Paragraph("Main entry point: <b>app.py</b>", S("Normal", fontSize=9, textColor=SLATE, alignment=TA_CENTER)),
    HR(TEAL, 1.5),
    sp(40),
    Paragraph("April 2026  ·  CESNET-TimeSeries24  ·  Streamlit Cloud", sub_s),
]

# ── Page 2 — What was built ───────────────────────────────────────────────────
story += [
    Paragraph("", S("Normal", pageBreakBefore=1)),
    Paragraph("What Was Built", title_s),
    sp(4),
    HR(),
    sp(6),
    body("A <b>multi-model Streamlit dashboard</b> for Telecom ISPs that allows network engineers to "
         "upload hourly subnet traffic CSVs <i>once</i> and instantly analyse them across five "
         "AI/ML models — all from a single browser window, no coding required."),
    sp(10),
    heading1("Project File Structure"),
    HR(SLATE, 0.5),
]

tree = [
    ("File / Folder", "Purpose"),
    ("app.py",                         "Home page — hero banner, 5 model cards, how-to guide"),
    ("pages/upload.py",                "CSV upload — parses file, stores in shared session state"),
    ("pages/member1_model.py",         "🔴 SLA Violation Risk — XGBoost inference (pre-trained)"),
    ("pages/member2_model.py",         "🔮 Time-Series Forecasting — Holt-Winters ETS"),
    ("pages/member3_model.py",         "🚨 Anomaly Detection — Isolation Forest (on-the-fly)"),
    ("pages/member4_model.py",         "🔵 Clustering & Root Cause — K-Means + DBSCAN"),
    ("pages/member5_model.py",         "📊 Correlation & Dependency — Pearson/Spearman + network graph"),
    ("utils/styles.py",                "Shared CSS, inject_css(), page_header() helpers"),
    ("utils/feature_engineering.py",   "parse_csv, engineer_features, prepare_for_model"),
    ("utils/api_client.py",            "call_endpoint() — FastAPI-ready HTTP client stub"),
    ("sla_xgboost_model.json",         "Pre-trained XGBoost model weights (replace with real file)"),
    ("model_config.pkl",               "Scaler + feature_cols + optimal_threshold (replace with real)"),
    ("requirements.txt",               "All Python dependencies"),
    (".streamlit/config.toml",         "Dark theme + 200 MB upload limit"),
]
def make_tree_row(idx, row):
    if idx == 0:
        return [Paragraph(c, S("Normal", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, leading=11)) for c in row]
    return [Paragraph(row[0], code_s), Paragraph(row[1], body_s)]

story.append(styled_table(
    [make_tree_row(idx, row) for idx, row in enumerate(tree)],
    [5.5*cm, 11*cm],
))

story += [
    sp(12),
    heading1("How the App Works"),
    HR(SLATE, 0.5),
    body("1. User goes to <b>Upload Data</b> in the sidebar and uploads a CSV once."),
    sp(3),
    body("2. The CSV is parsed, datetime-indexed, and stored in <b>st.session_state[\"df\"]</b> — "
         "shared automatically across every model page."),
    sp(3),
    body("3. User navigates to any of the 5 model pages. Each page reads the shared data, "
         "runs its own analysis, and shows interactive Plotly charts."),
    sp(3),
    body("4. Every page has a <b>Download CSV</b> button to export results."),
]

# ── Page 3 — Models ───────────────────────────────────────────────────────────
story.append(Paragraph("", S("Normal", pageBreakBefore=1)))
story += [
    Paragraph("The Five AI Models", title_s),
    sp(4),
    HR(),
]

models = [
    {
        "num": "01", "icon": "🔮", "name": "Time-Series Forecasting",
        "color": BLUE, "page": "pages/member2_model.py",
        "bos": "BO1, BO3, BO7",
        "purpose": "Predict future traffic KPIs (n_bytes, n_flows, etc.)",
        "algo": "Holt-Winters Exponential Smoothing (ETS) with additive/multiplicative trend and seasonality",
        "output": "Hourly forecast with 95% confidence intervals + STL decomposition (trend, seasonal, residual)",
        "controls": "Forecast horizon (1–168 h), seasonal period (24 h / 168 h), trend & seasonality type",
        "needs": "No pre-trained file needed — fits on uploaded data automatically",
    },
    {
        "num": "02", "icon": "🚨", "name": "Anomaly Detection",
        "color": AMBER, "page": "pages/member3_model.py",
        "bos": "BO1, BO4, BO7",
        "purpose": "Detect abnormal network behaviour without any labelled data",
        "algo": "Isolation Forest (sklearn) trained in-browser on uploaded data",
        "output": "Normalised anomaly score per hour, flagged events table, PCA scatter of normal vs anomaly",
        "controls": "Contamination fraction (0.01–0.20), number of trees, rolling feature toggle",
        "needs": "No pre-trained file needed — fits on uploaded data automatically",
    },
    {
        "num": "03", "icon": "🔴", "name": "SLA Violation Risk",
        "color": RED, "page": "pages/member1_model.py",
        "bos": "BO1, BO4",
        "purpose": "Predict the probability of an SLA breach for each hour",
        "algo": "Pre-trained XGBoost binary classifier with optimal decision threshold",
        "output": "Breach probability (0–1), risk timeline, breach-hours table, daily breach bar chart",
        "controls": "Reads threshold from model_config.pkl — no manual tuning needed",
        "needs": "⚠️  Requires: sla_xgboost_model.json + model_config.pkl in repo root",
    },
    {
        "num": "04", "icon": "🔵", "name": "Clustering & Root Cause",
        "color": PURPLE, "page": "pages/member4_model.py",
        "bos": "BO2, BO4, BO6",
        "purpose": "Segment traffic hours into behavioural groups; localise fault patterns",
        "algo": "K-Means or DBSCAN on StandardScaler + rolling-stats features, visualised with PCA 2-D",
        "output": "Cluster label per hour (auto-named: Normal / Congested / Spiky / Low-traffic), cluster profiles table",
        "controls": "Algorithm choice, k (K-Means) or eps + min_samples (DBSCAN)",
        "needs": "No pre-trained file needed — fits on uploaded data automatically",
    },
    {
        "num": "05", "icon": "📊", "name": "Correlation & Dependency",
        "color": TEAL, "page": "pages/member5_model.py",
        "bos": "BO2",
        "purpose": "Understand which KPIs move together; identify root-cause relationships",
        "algo": "Pearson / Spearman static correlation, rolling windowed correlation, NetworkX dependency graph",
        "output": "Heatmap, top-pairs bar chart, rolling correlation timeline, interactive network graph",
        "controls": "Correlation method, rolling window (6–168 h), network edge threshold",
        "needs": "No pre-trained file needed — computed directly from uploaded data",
    },
]

for m in models:
    story += [
        sp(8),
        Paragraph(f"{m['icon']}  Model {m['num']} — {m['name']}", h2_s),
        HR(m["color"], 0.6),
    ]
    rows = [
        ["File", m["page"]],
        ["Covers (BOs)", m["bos"]],
        ["Purpose", m["purpose"]],
        ["Algorithm", m["algo"]],
        ["Output", m["output"]],
        ["Sidebar controls", m["controls"]],
        ["What teammate needs to provide", m["needs"]],
    ]
    t = Table(
        [[Paragraph(r[0], S("Normal", fontSize=8, fontName="Helvetica-Bold", textColor=SLATE, leading=11)),
          Paragraph(r[1], S("Normal", fontSize=8, leading=12))] for r in rows],
        colWidths=[4.2*cm, 12.3*cm],
    )
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f8fafc"), WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story.append(t)

# ── Page 4 — Teammate guide ───────────────────────────────────────────────────
story.append(Paragraph("", S("Normal", pageBreakBefore=1)))
story += [
    Paragraph("What Each Teammate Needs to Provide", title_s),
    sp(4),
    HR(),
    sp(6),
    body("Each model page is already coded and working. What teammates need to do is provide "
         "their <b>trained model files</b> (if applicable) and test with real data. "
         "Models that train on-the-fly need nothing extra."),
    sp(10),
]

contrib = [
    ["Teammate", "Model", "Files to provide", "Where to drop them"],
    ["Member 1\n(your model)", "SLA Violation Risk\n(XGBoost)", "sla_xgboost_model.json\nmodel_config.pkl\n\nmodel_config.pkl must contain:\n• scaler (StandardScaler)\n• feature_cols (list)\n• optimal_threshold (float)", "Repo root /"],
    ["Member 2", "Time-Series\nForecasting", "No model file needed.\nApp fits Holt-Winters on\nthe uploaded CSV.", "—"],
    ["Member 3", "Anomaly\nDetection", "No model file needed.\nApp trains Isolation Forest\non the uploaded CSV.", "—"],
    ["Member 4", "Clustering &\nRoot Cause", "No model file needed.\nApp runs K-Means / DBSCAN\non the uploaded CSV.", "—"],
    ["Member 5", "Correlation &\nDependency", "No model file needed.\nAll stats computed live\nfrom uploaded CSV.", "—"],
]

story.append(styled_table(
    [[Paragraph(c, S("Normal", fontSize=8,
                     fontName="Helvetica-Bold" if i==0 else "Helvetica",
                     textColor=WHITE if i==0 else colors.black,
                     leading=12)) for c in row]
     for i, row in enumerate(contrib)],
    [2.5*cm, 3*cm, 7.5*cm, 3.5*cm],
    header_color=NAVY,
))

story += [
    sp(14),
    heading1("CSV Format Expected"),
    HR(SLATE, 0.5),
    body("Every model page expects the same CSV format:"),
    sp(4),
]

csv_rows = [
    ["Requirement", "Detail"],
    ["Datetime column", "Any column with timestamps (auto-detected). Name it 'timestamp', 'time', 'datetime', etc."],
    ["Numeric columns", "One or more traffic metric columns: n_bytes, n_flows, packets, bytes_in, bytes_out, etc."],
    ["Frequency", "Hourly resolution is required for rolling-24h and lag features to make sense."],
    ["Minimum rows", "≥ 48 rows for forecasting; ≥ 25 rows for anomaly/clustering; any for correlation."],
    ["Missing values", "Allowed — auto-imputed with column median before model inference."],
]
story.append(styled_table(
    [[Paragraph(c, S("Normal", fontSize=8,
                     fontName="Helvetica-Bold" if i==0 else "Helvetica",
                     textColor=WHITE if i==0 else colors.black, leading=12)) for c in row]
     for i, row in enumerate(csv_rows)],
    [4*cm, 12.5*cm],
))

story += [
    sp(4),
    body("Example first few lines of a valid CSV:"),
    sp(3),
    code("timestamp,bytes_in,bytes_out,packets"),
    code("2024-01-01 00:00:00,1048576,819200,4800"),
    code("2024-01-01 01:00:00,983040,753664,4512"),
    code("2024-01-01 02:00:00,1126400,880640,5120"),
    code("..."),
    sp(14),
    heading1("Feature Engineering (SLA model — auto-applied)"),
    HR(SLATE, 0.5),
    body("The SLA page automatically engineers these features before running XGBoost inference:"),
    sp(4),
]

feat_rows = [
    ["Feature group", "What it creates", "Example"],
    ["Z-scores",        "Global z-score per column",                       "{col}_zscore"],
    ["Rolling 24h",     "mean, std, min, max over last 24 hours",          "{col}_roll24_mean, _std, _min, _max"],
    ["Lag features",    "Values 1 hour ago, 6 hours ago, 24 hours ago",    "{col}_lag1, _lag6, _lag24"],
    ["Time features",   "Hour of day, day of week, weekend flag, holiday", "hour, dayofweek, is_weekend, is_holiday"],
]
story.append(styled_table(
    [[Paragraph(c, S("Normal", fontSize=8,
                     fontName="Helvetica-Bold" if i==0 else "Helvetica",
                     textColor=WHITE if i==0 else colors.black, leading=12)) for c in row]
     for i, row in enumerate(feat_rows)],
    [3.5*cm, 6.5*cm, 6.5*cm],
))

# ── Page 5 — Deployment & links ───────────────────────────────────────────────
story.append(Paragraph("", S("Normal", pageBreakBefore=1)))
story += [
    Paragraph("Deployment & Quick-Start", title_s),
    sp(4),
    HR(),
    sp(6),
    heading1("Deploy to Streamlit Cloud"),
    HR(SLATE, 0.5),
    body("1. Go to <b>share.streamlit.io</b> and sign in with GitHub."),
    sp(3),
    body("2. Click <b>New app</b> and fill in:"),
    sp(2),
    code("Repository :  3morad/Sla-breach"),
    code("Branch     :  claude/xgboost-sla-streamlit-app-h10XP"),
    code("Main file  :  app.py"),
    sp(3),
    body("3. Click <b>Deploy</b> — Streamlit Cloud installs requirements.txt automatically "
         "and gives you a public URL within ~2 minutes."),
    sp(3),
    body("4. To update: push changes to the branch → Streamlit Cloud auto-redeploys."),
    sp(14),
    heading1("Run Locally"),
    HR(SLATE, 0.5),
    code("git clone https://github.com/3morad/Sla-breach"),
    code("cd Sla-breach"),
    code("git checkout claude/xgboost-sla-streamlit-app-h10XP"),
    code("pip install -r requirements.txt"),
    code("streamlit run app.py"),
    sp(14),
    heading1("Add Your Real Model Files (Member 1 / SLA)"),
    HR(SLATE, 0.5),
    body("Replace the placeholder model files with your real trained files, then push:"),
    sp(4),
    code("# Drop your files into the repo root, then:"),
    code("git add sla_xgboost_model.json model_config.pkl"),
    code('git commit -m "Replace placeholder model with production artefacts"'),
    code("git push"),
    sp(3),
    body("Streamlit Cloud detects the push and redeploys automatically."),
    sp(14),
    heading1("Key Links"),
    HR(SLATE, 0.5),
    sp(4),
]

links = [
    ["Resource", "URL"],
    ["GitHub Repo",       "https://github.com/3morad/Sla-breach"],
    ["Branch",            "claude/xgboost-sla-streamlit-app-h10XP"],
    ["Streamlit Cloud",   "https://share.streamlit.io"],
    ["CESNET Dataset",    "https://zenodo.org/records/10958829"],
    ["XGBoost docs",      "https://xgboost.readthedocs.io"],
    ["Streamlit docs",    "https://docs.streamlit.io"],
]
story.append(styled_table(
    [[Paragraph(c, S("Normal", fontSize=9,
                     fontName="Helvetica-Bold" if i==0 else "Helvetica",
                     textColor=WHITE if i==0 else BLUE if i > 0 and j == 1 else colors.black,
                     leading=13)) for j, c in enumerate(row)]
     for i, row in enumerate(links)],
    [5*cm, 11.5*cm],
))

story += [
    sp(30),
    HR(TEAL, 1),
    sp(6),
    Paragraph("NetPulse Analytics · Built on CESNET-TimeSeries24 · April 2026", sub_s),
]

doc.build(story)
print("PDF generated: NetPulse_Dashboard_Overview.pdf")
