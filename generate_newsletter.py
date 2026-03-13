#!/usr/bin/env python3
# Ford Raptor Price Trend Report Generator
# (Cleaned: no unescaped Windows paths in docstrings; uses raw strings for path defaults.)

import os
import re
import sys
import csv
import math
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median, mean
from xml.sax.saxutils import escape

import numpy as np
import matplotlib.pyplot as plt

import openai
import git
from git import GitCommandError
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

# ----------------------------
# Config & paths (raw strings)
# ----------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_env = _SCRIPT_DIR / ".env"

# preserve existing env vars, otherwise load .env
if _env.exists():
    with open(_env) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)
else:
    load_dotenv(dotenv_path=_env)

API_KEY = os.environ.get("PROJECT_API_KEY", "")
if not API_KEY:
    print("Missing or empty PROJECT_API_KEY in environment/.env — please add it and re-run.")
    sys.exit(1)

# Local repo path (raw)
REPO_PATH = Path(os.environ.get("REPO_PATH", r"C:\Users\jpichett\Raptor-Report"))
# Output folder exactly as you requested
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(REPO_PATH / "newsletters")))

GIT_REMOTE_URL = os.environ.get("GIT_REMOTE_URL", "https://github.com/AMD-JP/Raptor-Report.git")
GIT_REMOTE_NAME = os.environ.get("GIT_REMOTE_NAME", "origin")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")

DATA_CSV = os.environ.get("DATA_CSV", str(_SCRIPT_DIR / "data" / "prices.csv"))

MODEL = "GPT-oss-20B"
MAX_TOKENS = 3500
TEMPERATURE = 0.35

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("raptor_report")

FORD_BLUE = colors.HexColor("#003478")
GREY = colors.HexColor("#666666")
BORDER = colors.HexColor("#DDDDDD")

# ----------------------------
# OpenAI / AMD client
# ----------------------------

def make_client():
    return openai.OpenAI(
        base_url="https://llm-api.amd.com/OnPrem",
        api_key=API_KEY,
        default_headers={
            "Ocp-Apim-Subscription-Key": API_KEY,
            "user": "raptor-price-bot",
        },
    )

# ----------------------------
# Helpers (sanitize, LLM call)
# ----------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[#*`]", "", text)
    for a,b in [("\u2014","-"),("\u2013","-"),("\u2018","'"),("\u2019","'"),("\u201c",'"'),("\u201d",'"')]:
        text = text.replace(a,b)
    text = text.encode("ascii","replace").decode("ascii")
    return escape(text).strip()

def call_llm(client, prompt, section, max_tokens=None):
    log.info("Generating %s", section)
    system = (
        "You are an automotive market analyst. Be concise but specific. "
        "Always explicitly label whether statements refer to NEW (MSRP) or USED (asking price) when applicable."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=(max_tokens or MAX_TOKENS),
            temperature=TEMPERATURE,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":prompt},
            ],
        )
    except Exception as e:
        msg = str(e).lower()
        if any(tok in msg for tok in ("401", "access denied", "invalid subscription", "authentication")):
            log.error("LLM authentication error: %s", e)
            raise RuntimeError(
                "LLM authentication failed (401). Check PROJECT_API_KEY in your .env and confirm the key is valid."
            ) from e
        log.error("LLM call failed: %s", e)
        raise
    content = response.choices[0].message.content
    return clean_text(content)

# ----------------------------
# CSV parsing & aggregation
# ----------------------------

def parse_row(row):
    try:
        date_str = row.get("date") or row.get("sale_date") or row.get("listing_date")
        date = datetime.fromisoformat(date_str).date() if date_str else None
        condition = (row.get("condition") or row.get("status") or "").strip().lower()
        msrp = float(row.get("msrp")) if row.get("msrp") else None
        price = float(row.get("price") or row.get("asking_price") or row.get("sale_price")) if (row.get("price") or row.get("asking_price") or row.get("sale_price")) else None
        region = row.get("region") or row.get("state") or row.get("location") or "Unknown"
        model_year = int(row.get("model_year")) if row.get("model_year") else (date.year if date else None)
        return {"date": date, "condition": condition, "msrp": msrp, "price": price, "region": region, "model_year": model_year}
    except Exception:
        return None

def load_and_aggregate(csv_path):
    rows = []
    csv_file = Path(csv_path)
    if not csv_file.exists():
        log.info("CSV not found at %s", csv_file)
        return None
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = parse_row(r)
            if parsed:
                rows.append(parsed)
    if not rows:
        log.info("CSV present but no parsable rows")
        return None

    msrp_by_year_new = {}
    price_by_year_used = {}
    now_year = datetime.now().year

    date_years = [r["date"].year for r in rows if r["date"]]
    years = set(date_years) if date_years else set([now_year])
    latest_year = max(date_years) if date_years else now_year

    for r in rows:
        yr = r["date"].year if r["date"] else (r["model_year"] or now_year)
        cond = r["condition"] or "used"
        if cond == "new" and r["msrp"]:
            msrp_by_year_new.setdefault(yr, []).append(r["msrp"])
        if cond == "used" and r["price"] is not None:
            price_by_year_used.setdefault(yr, []).append(r["price"])

    region_prices = {}
    for r in rows:
        r_year = r["date"].year if r["date"] else (r["model_year"] or now_year)
        if r_year == latest_year and r["price"] is not None:
            region_prices.setdefault(r["region"], []).append(r["price"])
    region_prices_latest = {reg: mean(vals) for reg, vals in region_prices.items()}

    markups = []
    markup_pct = []
    dep_by_model_year = {}
    for r in rows:
        if r["msrp"] and r["price"] is not None:
            mu = r["price"] - r["msrp"]
            markups.append(mu)
            if r["msrp"] != 0:
                markup_pct.append(mu / r["msrp"] * 100)
            if r["condition"] == "used" and r["model_year"]:
                dep = (r["msrp"] - r["price"]) / r["msrp"] * 100 if r["msrp"] else None
                if dep is not None:
                    dep_by_model_year.setdefault(r["model_year"], []).append(dep)
    dealer_markup_stats = {
        "count": len(markups),
        "avg_markup": mean(markups) if markups else 0.0,
        "median_markup": median(markups) if markups else 0.0,
        "avg_markup_pct": mean(markup_pct) if markup_pct else 0.0
    }
    depreciation_by_model_year = {my: mean(vals) for my, vals in dep_by_model_year.items()}

    years_sorted = sorted(years)
    msrp_new_series = [median(msrp_by_year_new[y]) if y in msrp_by_year_new and msrp_by_year_new[y] else None for y in years_sorted]
    price_used_series = [median(price_by_year_used[y]) if y in price_by_year_used and price_by_year_used[y] else None for y in years_sorted]
    msrp_new_series = [float(v) if v is not None else float("nan") for v in msrp_new_series]
    price_used_series = [float(v) if v is not None else float("nan") for v in price_used_series]

    kpis = {
        "latest_year": latest_year,
        "median_new_msrp_latest": float(median(msrp_by_year_new[latest_year]) if latest_year in msrp_by_year_new else float("nan")),
        "median_used_price_latest": float(median(price_by_year_used[latest_year]) if latest_year in price_by_year_used else float("nan")),
    }

    return {
        "years": years_sorted,
        "msrp_new": msrp_new_series,
        "price_used": price_used_series,
        "regions_latest": region_prices_latest,
        "dealer_markup": dealer_markup_stats,
        "depreciation_by_model_year": depreciation_by_model_year,
        "kpis": kpis,
        "competition": {"Ram TRX": 28, "Chevy ZR2": 22, "Toyota TRD Pro": 18, "Other": 32}
    }

# ----------------------------
# Synthetic fallback & charts
# ----------------------------

def synth_data():
    years = list(range(2017, 2026))
    base = np.array([55000 + (y - 2017) * 1500 for y in years], dtype=float)
    rng = np.random.default_rng(2026)
    noise = rng.normal(0, 800, size=base.shape)
    msrp_new = (base + noise).round(-2)
    price_used = (msrp_new * 0.82 + rng.normal(0, 1500, size=base.shape)).round(-2)
    regions = {"Texas": (msrp_new[-1]*1.02).round(-2), "California": (msrp_new[-1]*1.08).round(-2), "Midwest": (msrp_new[-1]*0.97).round(-2)}
    latest_median = float(np.median(msrp_new[-3:]).round(2))
    return {
        "years": years,
        "msrp_new": msrp_new.tolist(),
        "price_used": price_used.tolist(),
        "regions_latest": regions,
        "competition": {"Ram TRX": 28, "Chevy ZR2": 22, "Toyota TRD Pro": 18, "Other": 32},
        "dealer_markup": {"count":0,"avg_markup":0.0,"median_markup":0.0,"avg_markup_pct":0.0},
        "depreciation_by_model_year": {},
        "kpis": {"latest_year": years[-1], "median_new_msrp_latest": latest_median, "median_used_price_latest": float(median(price_used[-3:]).round(2))}
    }

def make_charts(data, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    imgs = {}
    years = data["years"]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(years, data.get("msrp_new", [float("nan")]*len(years)), marker="o", label="Median MSRP (New)")
    ax.plot(years, data.get("price_used", [float("nan")]*len(years)), marker="o", linestyle="--", label="Median Asking (Used)")
    ax.set_title("Median MSRP (New) vs Median Asking Price (Used) by Year")
    ax.set_xlabel("Year"); ax.set_ylabel("USD"); ax.legend(fontsize=8); ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    p1 = outdir / "msrp_vs_used_trend.png"; fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig); imgs["msrp_vs_used_trend"] = str(p1)
    regions = data.get("regions_latest", {})
    if regions:
        fig, ax = plt.subplots(figsize=(6,3)); ax.bar(list(regions.keys()), list(regions.values())); ax.set_title(f"Regional Average Asking Price ({data.get('kpis',{}).get('latest_year','')})"); ax.set_ylabel("USD")
        p2 = outdir / "regional_bar.png"; fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig); imgs["regional_bar"] = str(p2)
    comp = data.get("competition", {})
    if comp:
        fig, ax = plt.subplots(figsize=(6,3)); labels = list(comp.keys()); vals = list(comp.values()); ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=140); ax.set_title("Relative Competitive Share")
        p3 = outdir / "competition_pie.png"; fig.tight_layout(); fig.savefig(p3, dpi=150); plt.close(fig); imgs["competition_pie"] = str(p3)
    return imgs

# ----------------------------
# PDF builder (charts at end)
# ----------------------------

def styles():
    return {
        "title": ParagraphStyle("title", fontSize=24, fontName="Helvetica-Bold", textColor=colors.white, alignment=TA_CENTER),
        "section": ParagraphStyle("section", fontSize=14, fontName="Helvetica-Bold", textColor=FORD_BLUE, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("body", fontSize=10, leading=14, alignment=TA_JUSTIFY),
        "kpi": ParagraphStyle("kpi", fontSize=10, leading=12, alignment=TA_CENTER),
        "footer": ParagraphStyle("footer", fontSize=8, textColor=GREY, alignment=TA_CENTER)
    }

def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def build_pdf(sections, data, imgs, output, date):
    s = styles()
    doc = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.5*inch, bottomMargin=0.75*inch, title=f"Ford Raptor Price Report - {date}", author="Raptor Price Tracker Bot")
    story = []; width = letter[0] - 1.5*inch
    masthead = Table([[Paragraph("FORD RAPTOR", s["title"])],[Paragraph("PRICE TREND REPORT", s["title"])],[Paragraph(date, s["footer"])]], colWidths=[width]); masthead.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),FORD_BLUE), ("TOPPADDING",(0,0),(-1,-1),12), ("BOTTOMPADDING",(0,0),(-1,-1),12)])); story.append(masthead); story.append(Spacer(1,10))
    kpis = data.get("kpis", {}); kpi_table = Table([[Paragraph("Latest Year", s["kpi"]), Paragraph("Median MSRP (New)", s["kpi"]), Paragraph("Median Asking (Used)", s["kpi"])],[str(kpis.get("latest_year","")), f"${kpis.get('median_new_msrp_latest', float('nan')):,.0f}" if not math.isnan(kpis.get('median_new_msrp_latest', float('nan'))) else "N/A", f"${kpis.get('median_used_price_latest', float('nan')):,.0f}" if not math.isnan(kpis.get('median_used_price_latest', float('nan'))) else "N/A"]], colWidths=[width/3.0]*3); kpi_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F5F7FA")), ("ALIGN",(0,1),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("INNERGRID",(0,0),(-1,-1),0.25,BORDER), ("BOX",(0,0),(-1,-1),0.5,BORDER)])); story.append(kpi_table); story.append(Spacer(1,10))
    story.append(Paragraph("EXECUTIVE SUMMARY", s["section"])); story.append(HRFlowable(width=width, thickness=1, color=FORD_BLUE))
    for p in split_paragraphs(sections.get("summary","")): story.append(Paragraph(p, s["body"])); story.append(Spacer(1,8))
    sections_order = [("HISTORICAL PRICE TRENDS (NEW vs USED)","history"), ("USED MARKET ANALYSIS","used"), ("REGIONAL MARKET DIFFERENCES","regional"), ("COMPETITOR IMPACT","competition"), ("PRICE OUTLOOK (NEW / USED)","outlook")]
    for title,key in sections_order:
        story.append(Paragraph(title, s["section"])); story.append(HRFlowable(width=width, thickness=0.6, color=BORDER))
        for p in split_paragraphs(sections.get(key,""))[:4]:
            story.append(Paragraph(p, s["body"])); story.append(Spacer(1,8))
    story.append(Spacer(1,12)); story.append(HRFlowable(width=width, thickness=0.5, color=BORDER)); story.append(Spacer(1,8)); story.append(Paragraph("APPENDIX — CHARTS & VISUALS", s["section"])); story.append(HRFlowable(width=width, thickness=0.8, color=FORD_BLUE)); story.append(Spacer(1,8))
    if imgs.get("msrp_vs_used_trend"): story.append(Image(imgs["msrp_vs_used_trend"], width=width*0.95, height=3*inch)); story.append(Spacer(1,8))
    left = imgs.get("regional_bar"); right = imgs.get("competition_pie")
    if left and right:
        two_col = Table([[Image(left, width=width*0.47, height=2.2*inch), Image(right, width=width*0.47, height=2.2*inch)]], colWidths=[width*0.49, width*0.49]); two_col.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")])); story.append(two_col); story.append(Spacer(1,8))
    else:
        if left: story.append(Image(left, width=width*0.95, height=2.5*inch)); story.append(Spacer(1,8))
        if right: story.append(Image(right, width=width*0.95, height=2.5*inch)); story.append(Spacer(1,8))
    dep = data.get("depreciation_by_model_year", {})
    if dep:
        story.append(Paragraph("Depreciation by model year (avg % drop: MSRP -> Asking, USED)", s["section"]))
        rows = [["Model Year","Avg Depreciation %"]]
        for my in sorted(dep.keys()): rows.append([str(my), f"{dep[my]:.1f}%"])
        tbl = Table(rows, colWidths=[width*0.3, width*0.6]); tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,BORDER), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F5F7FA"))])); story.append(tbl); story.append(Spacer(1,8))
    story.append(HRFlowable(width=width, thickness=0.5, color=BORDER)); story.append(Paragraph(f"Ford Raptor Price Report | Generated {date}", styles()["footer"]))
    doc.build(story)

# ----------------------------
# Git helpers
# ----------------------------

def ensure_repo_present():
    try:
        repo = git.Repo(REPO_PATH)
        return repo
    except Exception:
        try:
            log.info("Local repo not found at %s — cloning from %s", REPO_PATH, GIT_REMOTE_URL)
            repo = git.Repo.clone_from(GIT_REMOTE_URL, REPO_PATH)
            log.info("Cloned repo to %s", REPO_PATH)
            return repo
        except Exception as e:
            log.warning("Failed to clone repo from %s: %s", GIT_REMOTE_URL, e)
            return None

def commit_and_push(file_path: Path) -> bool:
    repo = ensure_repo_present()
    if repo is None:
        log.warning("No repo available at %s and clone failed. Skipping upload.", REPO_PATH)
        return False
    try:
        remote = None
        try:
            remote = repo.remote(name=GIT_REMOTE_NAME)
            try:
                current_urls = list(remote.urls)
                if GIT_REMOTE_URL not in current_urls:
                    remote.set_url(GIT_REMOTE_URL)
                    log.info("Updated remote '%s' URL to %s", GIT_REMOTE_NAME, GIT_REMOTE_URL)
            except Exception as e:
                log.warning("Could not update remote URL: %s", e)
        except ValueError:
            try:
                repo.create_remote(GIT_REMOTE_NAME, GIT_REMOTE_URL)
                remote = repo.remote(name=GIT_REMOTE_NAME)
                log.info("Created remote '%s' -> %s", GIT_REMOTE_NAME, GIT_REMOTE_URL)
            except Exception as e:
                log.warning("Could not create remote: %s", e)
                remote = None
    except Exception as e:
        log.warning("Error ensuring remote: %s", e)
        remote = None
    try:
        repo.index.add([str(file_path.resolve())])
    except Exception as e:
        log.warning("Failed to add file to git index: %s", e)
        return False
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        repo.index.commit(f"raptor price report [{timestamp}]")
    except Exception as e:
        log.warning("Commit failed: %s", e)
        return False
    if remote is None:
        log.warning("No remote configured; skipping push.")
        return False
    try:
        push_info_list = remote.push(refspec=GIT_BRANCH)
        failed = False
        for info in push_info_list:
            if getattr(info, "error", None):
                log.warning("Push failed: %s", info.error); failed = True
            elif getattr(info, "summary", None):
                if "rejected" in info.summary.lower() or "failed" in info.summary.lower():
                    log.warning("Push summary indicated failure: %s", info.summary); failed = True
        if failed:
            log.warning("Push reported errors.")
            return False
        log.info("Push succeeded to %s/%s", GIT_REMOTE_NAME, GIT_BRANCH)
        return True
    except GitCommandError as e:
        log.warning("Git push command failed: %s", e)
        return False
    except Exception as e:
        log.warning("Unexpected error during git push: %s", e)
        return False

# ----------------------------
# Main
# ----------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%B %d, %Y")
    file_date = datetime.now().strftime("%Y-%m-%d")
    output = OUTPUT_DIR / f"raptor_price_report_{file_date}.pdf"
    client = make_client()
    aggregated = load_and_aggregate(DATA_CSV)
    if aggregated is None:
        log.info("Falling back to synthetic data (no CSV or no parsable rows).")
        aggregated = synth_data()
    sections = {}
    sections["summary"] = call_llm(client, f"Write a short executive summary (2 sentences) for a Ford Raptor price report dated {date}. Explicitly state whether you refer to NEW (MSRP) or USED (asking price). If available, reference the latest-year median MSRP for NEW and median asking price for USED from the dataset.", "Summary", max_tokens=160)
    sections["history"] = call_llm(client, "Give 3 short bullets comparing NEW (MSRP) and USED (asking) trends from 2017–2025 using the dataset. Be explicit when statements refer to NEW vs USED.", "Historical Prices", max_tokens=220)
    sections["used"] = call_llm(client, "Provide 3 concise bullets about the USED market: typical depreciation, average dealer markup behavior, and asking-price dynamics. Use dataset-derived stats where possible and label them clearly.", "Used Market", max_tokens=220)
    sections["regional"] = call_llm(client, "Provide 3 short bullets describing regional differences (Texas, California, Midwest). Specify whether the differences reference asking price (USED) or MSRP (NEW) and reference any regional averages present in the data.", "Regional Markets", max_tokens=180)
    sections["competition"] = call_llm(client, "Give 3 concise bullets on competitor impact (Ram TRX, Chevy ZR2, Toyota TRD Pro). Indicate how competitor supply/pricing has likely affected NEW vs USED Raptor pricing.", "Competition", max_tokens=200)
    sections["outlook"] = call_llm(client, "Provide a 3-bullet 12-month outlook separately for NEW (MSRP) and USED (asking) Raptor pricing. Label bullets NEW / USED.", "Outlook", max_tokens=180)
    tmpdir = Path(tempfile.mkdtemp(prefix="raptor_"))
    imgs = make_charts(aggregated, tmpdir)
    build_pdf(sections, aggregated, imgs, output, date)
    success = commit_and_push(output)
    if success:
        log.info("Upload succeeded.")
    else:
        log.warning("Upload failed; PDF saved locally to %s", output)
    log.info("Report generated: %s", output)

if __name__ == "__main__":
    main()
