"""
Ford Raptor Price Trend Report Generator

- Keeps original structure and AMD on-prem LLM client pattern.
- Loads .env (preserves original setdefault behavior).
- Default repository name set to a local folder named "Raptor-Report" (can be overridden via REPO_PATH env var).
- PDF output directory set to "<REPO_PATH>/Raptor Report" per your new file destination.
- Adds charts (MSRP trend, regional averages, competition) and a small KPI row for fast digestion.
- Preserves git commit & push to origin/main.
"""

import os
import re
import sys
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

# plotting & data
import numpy as np
import matplotlib.pyplot as plt

# LLM / git / env
import openai
import git
from dotenv import load_dotenv

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, Image
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent

_env = _SCRIPT_DIR / ".env"

# preserve original env-loading semantics:
if _env.exists():
    with open(_env) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)
else:
    # fallback to load_dotenv for safety
    load_dotenv(dotenv_path=_env)

# If you cloned the GitHub repo into a different path set REPO_PATH env var;
# default to a sibling/local directory called "Raptor-Report"
REPO_PATH = os.environ.get("REPO_PATH", str(_SCRIPT_DIR / "Raptor-Report"))

# Per your request, the PDF files go into a folder named "Raptor Report" inside the repo
OUTPUT_DIR = Path(REPO_PATH) / "Raptor Report"

API_KEY = os.environ.get("PROJECT_API_KEY", "")
GIT_REMOTE = "origin"
GIT_BRANCH = "main"

MODEL = "GPT-oss-20B"
MAX_TOKENS = 3500
TEMPERATURE = 0.4

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------

FORD_BLUE = colors.HexColor("#003478")
GREY = colors.HexColor("#666666")
BORDER = colors.HexColor("#DDDDDD")

# ---------------------------------------------------------------------
# AMD LLM CLIENT (UNCHANGED)
# ---------------------------------------------------------------------

def make_client():
    return openai.OpenAI(
        base_url="https://llm-api.amd.com/OnPrem",
        api_key="dummy",
        default_headers={
            "Ocp-Apim-Subscription-Key": API_KEY,
            "user": "raptor-price-bot",
        },
    )

# ---------------------------------------------------------------------
# TEXT SANITIZATION
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Sanitize LLM output so ReportLab never crashes."""
    if not text:
        return ""

    # remove html tags
    text = re.sub(r"<[^>]+>", "", text)

    # remove markdown
    text = re.sub(r"[#*`]", "", text)

    # normalize unicode punctuation
    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')

    # force ascii
    text = text.encode("ascii", "replace").decode("ascii")

    # escape HTML characters so ReportLab doesn't interpret them
    text = escape(text)

    return text.strip()

# ---------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------

def call_llm(client, prompt, section, max_tokens=None):
    log.info("Generating %s", section)

    system = (
        "You are an automotive market analyst specializing in pickup truck "
        "pricing and resale value trends."
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=(max_tokens or MAX_TOKENS),
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    return clean_text(content)

# ---------------------------------------------------------------------
# REPORT CONTENT
# ---------------------------------------------------------------------

def generate_content(client, date):
    sections = {}

    sections["summary"] = call_llm(client, f"""
Write a short executive summary for a Ford Raptor price report dated {date}.
Explain whether prices are trending up or down.
""", "Summary", max_tokens=140)

    sections["history"] = call_llm(client, f"""
Explain Ford Raptor MSRP changes from 2017 through 2025 and resale value trends.
Keep this to three short bullets or two short paragraphs.
""", "Historical Prices", max_tokens=220)

    sections["used"] = call_llm(client, f"""
Analyze used Ford Raptor prices across the United States including depreciation
and dealer markup behavior. Keep to three concise bullets.
""", "Used Market", max_tokens=220)

    sections["regional"] = call_llm(client, f"""
Explain regional Raptor price differences focusing on Texas, California,
and Midwest markets. Keep to 2-3 short bullets.
""", "Regional Markets", max_tokens=160)

    sections["competition"] = call_llm(client, f"""
Explain how competitor trucks affect Raptor pricing including Ram TRX,
Chevy Silverado ZR2 and Toyota TRD Pro. Keep to three concise bullets.
""", "Competition", max_tokens=200)

    sections["outlook"] = call_llm(client, f"""
Provide a short 12 month outlook for Ford Raptor pricing in 3 bullets.
""", "Outlook", max_tokens=160)

    return sections

# ---------------------------------------------------------------------
# SYNTHETIC DATA FOR CHARTS (self-contained)
# ---------------------------------------------------------------------

def synth_data():
    """Deterministic synthetic dataset for charts (2017-2025)."""
    years = list(range(2017, 2026))
    base = np.array([55000 + (y - 2017) * 1500 for y in years], dtype=float)
    rng = np.random.default_rng(2026)
    noise = rng.normal(0, 800, size=base.shape)
    msrp = (base + noise).round(-2)
    regions = {
        "Texas": (msrp * 1.02).round(-2),
        "California": (msrp * 1.08).round(-2),
        "Midwest": (msrp * 0.97).round(-2)
    }
    competition = {"Ram TRX": 28, "Chevy ZR2": 22, "Toyota TRD Pro": 18, "Other": 32}
    latest_median = float(np.median(msrp[-3:]).round(2))
    yoy = float(((msrp[-1] - msrp[-2]) / msrp[-2] * 100).round(2))
    return {"years": years, "msrp": msrp.tolist(), "regions": regions, "competition": competition, "kpis": {"median_price": latest_median, "yoy_change_pct": yoy}}

# ---------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------

def make_charts(data, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    imgs = {}

    # MSRP trend (line)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(data["years"], data["msrp"], marker="o", linewidth=2)
    ax.set_title("Ford Raptor MSRP — Median (2017–2025)", fontsize=10)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("MSRP (USD)", fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    p1 = outdir / "msrp_trend.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    imgs["msrp_trend"] = str(p1)

    # Regional bar (latest)
    latest = {k: float(v[-1]) for k, v in data["regions"].items()}
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(latest.keys()), list(latest.values()))
    ax.set_title("Regional Average Asking Price (latest year)", fontsize=10)
    ax.set_ylabel("Price (USD)", fontsize=8)
    fig.tight_layout()
    p2 = outdir / "regional_bar.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    imgs["regional_bar"] = str(p2)

    # Competition pie
    labels = list(data["competition"].keys())
    vals = list(data["competition"].values())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=140)
    ax.set_title("Relative Competitive Share", fontsize=10)
    fig.tight_layout()
    p3 = outdir / "competition_pie.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    imgs["competition_pie"] = str(p3)

    return imgs

# ---------------------------------------------------------------------
# PDF STYLES
# ---------------------------------------------------------------------

def styles():
    return {
        "title": ParagraphStyle(
            "title",
            fontSize=24,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            alignment=TA_CENTER
        ),
        "section": ParagraphStyle(
            "section",
            fontSize=14,
            fontName="Helvetica-Bold",
            textColor=FORD_BLUE,
            spaceBefore=12,
            spaceAfter=6
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10,
            leading=16,
            alignment=TA_JUSTIFY
        ),
        "kpi": ParagraphStyle(
            "kpi",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER
        ),
        "footer": ParagraphStyle(
            "footer",
            fontSize=8,
            textColor=GREY,
            alignment=TA_CENTER
        )
    }

# ---------------------------------------------------------------------
# PARAGRAPH SPLITTER
# ---------------------------------------------------------------------

def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

# ---------------------------------------------------------------------
# PDF BUILDER (embeds charts + KPI table)
# ---------------------------------------------------------------------

def build_pdf(sections, data, imgs, output, date):
    s = styles()

    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.75 * inch,
        title=f"Ford Raptor Price Report - {date}",
        author="Raptor Price Tracker Bot",
    )

    story = []
    width = letter[0] - 1.5 * inch

    # Masthead
    masthead = Table([
        [Paragraph("FORD RAPTOR", s["title"])],
        [Paragraph("PRICE TREND REPORT", s["title"])],
        [Paragraph(date, s["footer"])]
    ], colWidths=[width])
    masthead.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), FORD_BLUE),
        ("TOPPADDING",(0,0),(-1,-1),16),
        ("BOTTOMPADDING",(0,0),(-1,-1),16)
    ]))
    story.append(masthead)
    story.append(Spacer(1,12))

    # KPI row
    kpis = data.get("kpis", {})
    kpi_table = Table([
        [Paragraph("Median (recent)", s["kpi"]), Paragraph("YoY %", s["kpi"]), Paragraph("Latest Year", s["kpi"])],
        [f"${kpis.get('median_price',0):,.0f}", f"{kpis.get('yoy_change_pct',0):+.2f}%", str(data["years"][-1])]
    ], colWidths=[width/3.0]*3)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F5F7FA")),
        ("ALIGN",(0,1),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("INNERGRID",(0,0),(-1,-1),0.25,BORDER),
        ("BOX",(0,0),(-1,-1),0.5,BORDER),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1,10))

    # Executive summary
    story.append(Paragraph("EXECUTIVE SUMMARY", s["section"]))
    story.append(HRFlowable(width=width, thickness=1, color=FORD_BLUE))
    for p in split_paragraphs(sections.get("summary","")):
        story.append(Paragraph(p, s["body"]))
    story.append(Spacer(1,10))

    # Key visuals
    story.append(Paragraph("KEY VISUALS", s["section"]))
    story.append(HRFlowable(width=width, thickness=0.8, color=FORD_BLUE))
    story.append(Spacer(1,6))
    story.append(Image(imgs["msrp_trend"], width=width*0.95, height=3*inch))
    story.append(Spacer(1,8))
    two_col = Table([[Image(imgs["regional_bar"], width=width*0.47, height=2.0*inch),
                      Image(imgs["competition_pie"], width=width*0.47, height=2.0*inch)]],
                    colWidths=[width*0.49, width*0.49])
    two_col.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(two_col)
    story.append(Spacer(1,12))

    # Short analytic sections (limit to 3 short paragraphs each)
    sections_order = [
        ("HISTORICAL PRICE TRENDS","history"),
        ("USED MARKET ANALYSIS","used"),
        ("REGIONAL MARKET DIFFERENCES","regional"),
        ("COMPETITOR IMPACT","competition"),
        ("PRICE OUTLOOK","outlook")
    ]

    for title, key in sections_order:
        story.append(Paragraph(title, s["section"]))
        story.append(HRFlowable(width=width, thickness=0.6, color=BORDER))
        for p in split_paragraphs(sections.get(key,""))[:3]:
            story.append(Paragraph(p, s["body"]))
            story.append(Spacer(1,8))

    story.append(HRFlowable(width=width, thickness=0.5, color=BORDER))
    story.append(Paragraph(f"Ford Raptor Price Report | Generated {date}", s["footer"]))

    doc.build(story)

# ---------------------------------------------------------------------
# GIT PUSH
# ---------------------------------------------------------------------

def commit_and_push(file):
    repo = git.Repo(REPO_PATH)
    repo.git.add(str(file.resolve()))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repo.index.commit(f"raptor price report [{timestamp}]")
    repo.remote(name=GIT_REMOTE).push()

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    if not API_KEY:
        print("Missing PROJECT_API_KEY in .env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%B %d, %Y")
    file_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output = OUTPUT_DIR / f"raptor_price_report_{file_date}.pdf"

    client = make_client()

    # concise LLM-generated sections (keeps original prompts but prefers shorter responses)
    sections = generate_content(client, date)

    # synth data for charts (you can replace with real data ingestion later)
    data = synth_data()
    tmp_dir = Path(tempfile.mkdtemp(prefix="raptor_"))
    imgs = make_charts(data, tmp_dir)

    # build PDF with charts & KPIs
    build_pdf(sections, data, imgs, output, date)

    # commit & push to repo (assumes the repo exists at REPO_PATH and has a remote origin)
    commit_and_push(output)

    log.info("Report generated: %s", output)

if __name__ == "__main__":
    main()
