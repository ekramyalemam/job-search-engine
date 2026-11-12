#!/usr/bin/env python3
"""match_jobs.py - Phase 2: score every job in every results sheet against your resume.

Reads the latest / maybe / rejected sheets (CSV files, or tabs of a Google Sheet),
scores each job against resume/resume_match.txt, and writes jobs at or above the
threshold (default 75%) to a new sheet called "matched-List".

No LLM and no API key needed: scoring is rule-based, so it is free, repeatable and
safe to run in GitHub Actions. The method is adapted from Resume Matcher
(github.com/srbhr/resume-matcher, Apache-2.0): whole-word keyword matching
(services/text_match.py) and a weighted composite whose unscorable parts are
redistributed (services/ats.py).

Usage:
    python match_jobs.py                       # CSV mode, results/latest|maybe|rejected.csv
    python match_jobs.py --inputs a.csv b.csv  # explicit files
    python match_jobs.py --gsheet-id <ID>      # Google Sheet mode (needs GOOGLE_SERVICE_ACCOUNT_JSON)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- column mapping
COLMAP: dict[str, list[str]] = {
    "title": ["title", "job_title", "position", "job"],
    "company": ["company", "company_name", "employer"],
    "location": ["location", "job_location"],
    "description": ["description", "job_description", "desc", "details", "snippet"],
    "url": ["job_url", "job link", "url", "link", "job_url_direct", "apply_url"],
    "date": ["date_posted", "date posted", "posted", "date"],
    "site": ["site", "source", "board"],
    "reason": ["reason"],
}

OUT_COLS = [
    "rank", "match_pct", "title", "company", "location", "site", "date_posted",
    "source_sheet", "phase1_reason", "job_url", "missing_keywords", "strong_matches", "reason",
    "skills_pct", "title_pct", "level_pct", "credentials_pct",
]

STOP = {
    "a", "an", "and", "the", "of", "for", "in", "to", "at", "or", "with", "on", "-", "&",
    "senior", "sr", "jr", "junior", "lead", "head", "director", "vp", "vice", "president",
    "ii", "iii", "iv", "remote", "hybrid", "onsite", "on-site", "full", "time", "fulltime",
}


# ---------------------------------------------------------------- text helpers
def term_regex(names: list[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive match for any name; allows a plural 's'/'es'."""
    parts = sorted({re.escape(n.lower().strip()) for n in names if n and n.strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?:s|es)?(?!\w)", re.IGNORECASE)


def tokens(text: str) -> set[str]:
    out: set[str] = set()
    for w in re.findall(r"[a-z0-9+#/]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def pick(row: dict[str, str], key: str) -> str:
    for name in COLMAP[key]:
        if name in row and row[name] not in (None, ""):
            return str(row[name]).strip()
    return ""


# ---------------------------------------------------------------- resume + config
def load_resume_text(resume_path: str | Path, facts_path: str | Path = "confirmed_facts.yaml") -> tuple[str, dict[str, Any]]:
    """Resume text, plus any extra skills you confirmed in confirmed_facts.yaml (true facts only)."""
    text = Path(resume_path).read_text(encoding="utf-8")
    facts: dict[str, Any] = {}
    fp = Path(facts_path)
    if fp.exists():
        facts = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        extras = [s for lst in (facts.get("extra_skills") or {}).values() for s in (lst or [])]
        if extras:
            text += "\n\nConfirmed additional skills: " + ", ".join(extras) + "\n"
    return text, facts


class Matcher:
    def __init__(self, cfg: dict[str, Any], resume_text: str) -> None:
        self.cfg = cfg
        self.resume = resume_text
        self.resume_l = resume_text.lower()
        # term -> (group, regex)
        self.terms: dict[str, tuple[str, re.Pattern[str]]] = {}
        for group, items in cfg["terms"].items():
            for canon, aliases in items.items():
                self.terms[canon] = (group, term_regex([canon, *(aliases or [])]))
        self.creds = {c: term_regex([c, *(a or [])]) for c, a in cfg["credentials"].items()}
        self.resume_has = {t for t, (_, rx) in self.terms.items() if rx.search(self.resume)}
        self.resume_creds = {c for c, rx in self.creds.items() if rx.search(self.resume)}
        # role titles = the header line (2nd non-empty line) + short non-bullet lines naming a role
        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
        header = lines[1] if len(lines) > 1 else ""
        role_lines = [
            ln for ln in lines
            if len(ln) <= 90 and not ln.startswith(("-", "*")) and re.search(r"\b(Manager|Engineer|Director|Lead)\b", ln)
        ]
        self.title_tokens = tokens(" ".join([header, *role_lines]))
        self.body_tokens = tokens(resume_text)

    # ---- components
    def skills(self, jd: str) -> tuple[float | None, list[str], list[str], int]:
        found: dict[str, int] = {}
        for term, (_, rx) in self.terms.items():
            n = len(rx.findall(jd))
            if n:
                found[term] = min(n, 3)  # repeated mentions count, capped
        if not found:
            return None, [], [], 0
        total = sum(found.values())
        have = {t: w for t, w in found.items() if t in self.resume_has}
        missing = sorted((t for t in found if t not in self.resume_has), key=lambda t: -found[t])
        strong = sorted(have, key=lambda t: -have[t])
        return 100.0 * sum(have.values()) / total, strong, missing, len(found)

    def title_score(self, title: str, location: str, company: str) -> float | None:
        drop = tokens(location) | tokens(company)
        toks = tokens(title) - drop
        if not toks:
            return None
        pts = 0.0
        for t in toks:
            if t in self.title_tokens:
                pts += 1.0
            elif t in self.body_tokens:
                pts += 0.6
        return 100.0 * pts / len(toks)

    @staticmethod
    def level_from_title(title: str) -> float:
        t = title.lower()
        if re.search(r"\b(vice president|vp|svp|evp|chief|cto|coo|cio|president)\b", t) and "director" not in t:
            return 50.0
        if re.search(r"\b(director|head of|general manager|manager|senior manager|sr\.? manager)\b", t):
            return 100.0
        if re.search(r"\b(principal|lead|supervisor)\b", t):
            return 65.0
        if re.search(r"\b(engineer|analyst|specialist|coordinator|administrator|technician|developer|architect|consultant|associate|intern|officer|executive)\b", t):
            return 30.0
        return 60.0

    def level_score(self, title: str, jd: str) -> float:
        parts = [self.level_from_title(title)]
        mins = [int(m.group(1)) for m in re.finditer(r"(\d{1,2})\s*(?:\+|-\s*\d{1,2})?\s*\+?\s*(?:years|yrs|year)\b", jd.lower())]
        mins = [m for m in mins if 2 <= m <= 25]
        if mins:
            need = max(mins)
            have = self.cfg["resume_years"]
            parts.append(100.0 if need <= have else max(0.0, 100.0 - 10.0 * (need - have)))
        return sum(parts) / len(parts)

    def cred_score(self, jd: str) -> float | None:
        asked = [c for c, rx in self.creds.items() if rx.search(jd)]
        if not asked:
            return None
        return 100.0 * sum(c in self.resume_creds for c in asked) / len(asked)

    # ---- overall
    def score(self, job: dict[str, str]) -> dict[str, Any]:
        jd = job["description"]
        res: dict[str, Any] = {"status": "scored"}
        if len(jd) < self.cfg["min_description_chars"]:
            return {"status": "no_description", "match": 0.0}
        skills, strong, missing, n_found = self.skills(jd)
        comps = {
            "skills": skills,
            "title": self.title_score(job["title"], job["location"], job["company"]),
            "level": self.level_score(job["title"], jd),
            "credentials": self.cred_score(jd),
        }
        w = self.cfg["weights"]
        used = {k: v for k, v in comps.items() if v is not None}
        wsum = sum(w[k] for k in used)
        overall = sum(v * w[k] for k, v in used.items()) / wsum if wsum else 0.0
        if self.level_from_title(job["title"]) < self.cfg["target_level_min"]:
            res["status"] = "off_level"
        elif n_found < self.cfg["min_terms_for_confidence"]:
            res["status"] = "needs_review"
        res.update(match=round(overall, 1), comps=comps, strong=strong, missing=missing, n_found=n_found)
        res["reason"] = self.reason(res, job)
        return res

    @staticmethod
    def reason(r: dict[str, Any], job: dict[str, str]) -> str:
        c = r["comps"]
        bits = []
        if c["skills"] is not None:
            bits.append(f"covers {round(c['skills'])}% of the {r['n_found']} key terms in the posting")
        if c["title"] is not None:
            bits.append(f"title fit {round(c['title'])}")
        bits.append(f"level fit {round(c['level'])}")
        if c["credentials"] is not None:
            bits.append(f"credentials {round(c['credentials'])}")
        txt = "; ".join(bits).capitalize() + "."
        if r["status"] == "needs_review":
            txt += " Few known terms in the posting - review by hand."
        if r["status"] == "off_level":
            txt += " Title level is outside your Manager-Director target, so not listed."
        return txt


# ---------------------------------------------------------------- input / output
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [{(k or "").strip().lower(): (v or "") for k, v in row.items()} for row in csv.DictReader(f)]


def resolve_input(p: Path) -> Path:
    """Phase 1 names its files maybe_latest.csv / rejected_latest.csv; accept either spelling."""
    if p.exists():
        return p
    alt = p.with_name(p.stem + "_latest" + p.suffix)
    return alt if alt.exists() else p


def to_job(row: dict[str, str], sheet: str) -> dict[str, str]:
    return {k: pick(row, k) for k in COLMAP} | {"sheet": sheet}


def gspread_client() -> Any:
    import gspread  # lazy: only needed in Google Sheet mode

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not raw:
        sys.exit("Set GOOGLE_SERVICE_ACCOUNT_JSON (service-account key JSON) to use --gsheet-id.")
    return gspread.service_account_from_dict(json.loads(raw))


def read_gsheet(sheet_id: str, tabs: list[str]) -> dict[str, list[dict[str, str]]]:
    sh = gspread_client().open_by_key(sheet_id)
    out: dict[str, list[dict[str, str]]] = {}
    for tab in tabs:
        try:
            recs = sh.worksheet(tab).get_all_records()
        except Exception as e:  # missing tab: warn and continue
            print(f"  ! tab '{tab}' not read: {e}", file=sys.stderr)
            continue
        out[tab] = [{str(k).strip().lower(): str(v) for k, v in r.items()} for r in recs]
    return out


def write_gsheet(sheet_id: str, name: str, header: list[str], rows: list[list[Any]]) -> None:
    import gspread

    sh = gspread_client().open_by_key(sheet_id)
    try:
        ws = sh.worksheet(name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=max(len(rows) + 10, 100), cols=len(header))
    ws.update(values=[header, *rows], range_name="A1")


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="match_config.yaml")
    ap.add_argument("--resume", default="resume/resume_match.txt")
    ap.add_argument("--facts", default="confirmed_facts.yaml")
    ap.add_argument("--descriptions", help="CSV with 'Job link' + 'Description' (default: phase1/results/descriptions_latest.csv)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--inputs", nargs="*", help="CSV files to review (default: results/latest|maybe|rejected.csv)")
    ap.add_argument("--tabs", default="latest,maybe,rejected", help="Google Sheet tab names to review")
    ap.add_argument("--gsheet-id", default=os.environ.get("GSHEET_ID", ""))
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out-name", default="matched-List")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.threshold is not None:
        cfg["threshold"] = a.threshold
    if abs(sum(cfg["weights"].values()) - 1.0) > 1e-6:
        sys.exit("weights in match_config.yaml must add up to 1.0")
    resume_path = Path(a.resume)
    if not resume_path.exists() and Path(resume_path.name).exists():
        resume_path = Path(resume_path.name)   # flat layout: file sits next to the script
    if not resume_path.exists():
        sys.exit(f"Resume text not found: {resume_path}")
    resume_text, _ = load_resume_text(resume_path, a.facts)
    m = Matcher(cfg, resume_text)

    # 1) collect every job from every sheet, de-duplicated
    sheets: dict[str, list[dict[str, str]]] = {}
    if a.gsheet_id:
        sheets = read_gsheet(a.gsheet_id, [t.strip() for t in a.tabs.split(",")])
    else:
        files = [Path(p) for p in a.inputs] if a.inputs else [Path(a.results_dir) / f"{n.strip()}.csv" for n in a.tabs.split(",")]
        for p in files:
            p = resolve_input(p)
            if p.exists():
                sheets[p.stem.replace("_latest", "")] = read_csv_rows(p)
            else:
                print(f"  ! {p} not found, skipped", file=sys.stderr)
    if not sheets:
        sys.exit("No input sheets found.")

    # job descriptions live in their own Phase 1 file, joined to each job by its link
    desc_by_url: dict[str, str] = {}
    cands = [Path(a.descriptions)] if a.descriptions else [
        Path("phase1/results/descriptions_latest.csv"), Path(a.results_dir) / "descriptions_latest.csv", Path("descriptions_latest.csv")]
    for dp in cands:
        if dp.exists():
            for row in read_csv_rows(dp):
                u = (row.get("job link") or row.get("job_url") or row.get("url") or "").strip()
                if u and row.get("description"):
                    desc_by_url[u] = row["description"]
            print(f"Descriptions loaded: {len(desc_by_url)} from {dp}")
            break
    else:
        print("! No descriptions file found. Jobs without a description cannot be scored.", file=sys.stderr)

    jobs: dict[str, dict[str, str]] = {}
    seen_in: dict[str, list[str]] = {}
    for sheet, rows in sheets.items():
        for row in rows:
            j = to_job(row, sheet)
            if not j["title"]:
                continue
            if not j["description"] and j["url"] in desc_by_url:
                j["description"] = desc_by_url[j["url"]]
            key = j["url"] or "|".join([j["title"].lower(), j["company"].lower(), j["location"].lower()])
            if key not in jobs or len(j["description"]) > len(jobs[key]["description"]):
                jobs[key] = j
            seen_in.setdefault(key, [])
            if sheet not in seen_in[key]:
                seen_in[key].append(sheet)

    # 2) score
    results: list[tuple[dict[str, str], dict[str, Any]]] = []
    for key, j in jobs.items():
        j["sheets"] = "+".join(seen_in[key])
        results.append((j, m.score(j)))

    def out_row(j: dict[str, str], r: dict[str, Any], rank: int | str = "") -> list[Any]:
        c = r.get("comps", {})
        f = lambda v: "" if v is None else round(v)  # noqa: E731
        return [
            rank, r["match"], j["title"], j["company"], j["location"], j["site"], j["date"],
            j["sheets"], j["reason"], j["url"], ", ".join(r.get("missing", [])[:10]), ", ".join(r.get("strong", [])[:6]),
            r.get("reason", "No usable job description - cannot score."),
            f(c.get("skills")), f(c.get("title")), f(c.get("level")), f(c.get("credentials")),
        ]

    matched = sorted(
        [(j, r) for j, r in results if r["status"] == "scored" and r["match"] >= cfg["threshold"]],
        key=lambda x: -x[1]["match"],
    )
    allrows = sorted(results, key=lambda x: -x[1]["match"])
    m_rows = [out_row(j, r, i) for i, (j, r) in enumerate(matched, 1)]
    all_rows = [out_row(j, r) + [r["status"]] for j, r in allrows]

    # 3) write
    rdir = Path(a.results_dir)
    write_csv(rdir / f"{a.out_name}.csv", OUT_COLS, m_rows)
    write_csv(rdir / "scored_all.csv", OUT_COLS + ["status"], all_rows)
    if a.gsheet_id:
        write_gsheet(a.gsheet_id, a.out_name, OUT_COLS, m_rows)

    counts = {s: sum(1 for _, r in results if r["status"] == s) for s in ("scored", "needs_review", "no_description", "off_level")}
    near = sum(1 for j, r in results if r["status"] == "scored" and cfg["threshold"] - 10 <= r["match"] < cfg["threshold"])
    with_desc = len(results) - counts["no_description"]
    lines = [
        f"Reviewed {len(results)} unique jobs from {', '.join(sheets)} (de-duplicated).",
        f"Jobs that had a description to read: {with_desc} of {len(results)}",
        f"MATCHED >= {cfg['threshold']:.0f}%: {len(matched)}   | near miss (65-74%): {near}",
        f"Not listed: {counts['no_description']} without a usable description, {counts['needs_review']} need a manual look, {counts['off_level']} outside your target level.",
        "",
        "Top matches:",
        *[f"  {r['match']:>5}%  {j['title']} - {j['company']} ({j['location']})" for j, r in matched[:10]],
    ]
    summary = "\n".join(lines)
    (rdir / "match_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
