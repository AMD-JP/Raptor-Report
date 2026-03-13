"""
Ford Raptor Price Trend Report Generator
"""

import os
import re
import sys
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import matplotlib.pyplot as plt

import openai
import git
from dotenv import load_dotenv

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

if _env.exists():
    with open(_env) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)
else:
    load_dotenv(dotenv_path=_env)

API_KEY = os.environ.get("PROJECT_API_KEY", "")

# Git repo location
REPO_PATH = Path(r"C:\Users\jpichett\Raptor-Report")

# Output destination requested
OUTPUT_DIR = Path(r"C:\Users\jpichett\Raptor Report\newsletters")

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
# AMD LLM CLIENT
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

def clean_text(text: str):

    if not text:
        return ""

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[#*`]", "", text)

    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')

    text = text.encode("ascii", "replace").decode("ascii")

    text = escape(text)

    return text.strip()


# ---------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------

def call_llm(client, prompt, section):

    log.info("Generating %s", section)

    system = (
        "You are an automotive market analyst specializing in pickup truck "
        "pricing and resale value trends."
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
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
""", "Summary")

    sections["history"] = call_llm(client, f"""
Explain Ford Raptor MSRP changes from 2017 through 2025 and resale value trends.
""", "Historical Prices")

    sections["used"] = call_llm(client, f"""
Analyze used Ford Raptor prices across the United States including depreciation
and dealer markup behavior.
""", "Used Market")

    sections["regional"] = call_llm(client, f"""
Explain regional Raptor price differences focusing on Texas, California,
and Midwest markets.
""", "Regional Markets")

    sections["competition"] = call_llm(client, f"""
Explain how competitor trucks affect Raptor pricing including Ram TRX,
Chevy Silverado ZR2 and Toyota TRD Pro.
""", "Competition")

    sections["outlook"] = call_llm(client, f"""
Provide a short 12 month outlook for Ford Raptor pricing.
""", "Outlook")

    return sections


# ---------------------------------------------------------------------
# SAMPLE DATA FOR CHARTS
# ---------------------------------------------------------------------

def sample_data():

    years = list(range(2017, 2026))
    base = np.array([55000 + (y - 2017) * 1500 for y in years])
    noise = np.random.normal(0, 800, size=len(base))

    msrp = base + noise

    regions = {
        "Texas": msrp * 1.02,
        "California": msrp * 1.08,
        "Midwest": msrp * 0.97
    }

    competition = {
        "Ram TRX": 28,
        "Chevy ZR2": 22,
        "Toyota TRD Pro": 18,
        "Other": 32
    }

    return {
        "years": years,
        "msrp": msrp,
        "regions": regions,
        "competition": competition
    }


# ---------------------------------------------------------------------
# CHART GENERATION
# ---------------------------------------------------------------------

def make_charts(data, temp_dir):

    charts = {}

    # MSRP trend
    fig, ax = plt.subplots()
    ax.plot(data["years"], data["msrp"])
    ax.set_title("Raptor MSRP Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("MSRP")

    path = temp_dir / "msrp.png"
    fig.savefig(path)
    plt.close()
    charts["msrp"] = path

    # Regional prices
    latest = {k: v[-1] for k, v in data["regions"].items()}

    fig, ax = plt.subplots()
    ax.bar(list(latest.keys()), list(latest.values()))
    ax.set_title("Regional Price Comparison")

    path = temp_dir / "regions.png"
    fig.savefig(path)
    plt.close()
    charts["regions"] = path

    # Competition
    labels = list(data["competition"].keys())
    vals = list(data["competition"].values())

    fig, ax = plt.subplots()
    ax.pie(vals, labels=labels)

    path = temp_dir / "competition.png"
    fig.savefig(path)
    plt.close()
    charts["competition"] = path

    return charts


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
            spaceBefore=16,
            spaceAfter=6
        ),

        "body": ParagraphStyle(
            "body",
            fontSize=10,
            leading=16,
            alignment=TA_JUSTIFY
        ),

        "footer": ParagraphStyle(
            "footer",
            fontSize=8,
            textColor=GREY,
            alignment=TA_CENTER
        )
    }


# ---------------------------------------------------------------------
# PDF BUILDER
# ---------------------------------------------------------------------

def build_pdf(sections, charts, output, date):

    s = styles()

    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.75 * inch
    )

    story = []

    width = letter[0] - 1.5 * inch

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

    # charts
    story.append(Image(str(charts["msrp"]), width=width, height=200))
    story.append(Image(str(charts["regions"]), width=width, height=200))
    story.append(Image(str(charts["competition"]), width=width, height=200))

    story.append(Spacer(1,20))

    for title,key in [
        ("EXECUTIVE SUMMARY","summary"),
        ("HISTORICAL PRICE TRENDS","history"),
        ("USED MARKET ANALYSIS","used"),
        ("REGIONAL MARKET DIFFERENCES","regional"),
        ("COMPETITOR IMPACT","competition"),
        ("PRICE OUTLOOK","outlook")
    ]:

        story.append(Paragraph(title, s["section"]))
        story.append(HRFlowable(width=width, thickness=1, color=FORD_BLUE))
        story.append(Paragraph(sections[key], s["body"]))
        story.append(Spacer(1,12))

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

    sections = generate_content(client, date)

    data = sample_data()

    tmp = Path(tempfile.mkdtemp())

    charts = make_charts(data, tmp)

    build_pdf(sections, charts, output, date)

    commit_and_push(output)

    log.info("Report generated: %s", output)


if __name__ == "__main__":
    main()
