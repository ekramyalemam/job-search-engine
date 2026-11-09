"""Phase 1 (v3): search job boards and rank jobs by relevance (criteria.yaml)."""
import os
import random
import re
import time
import unicodedata
from datetime import date

import pandas as pd
import yaml
from jobspy import scrape_jobs


def clean(text):
    """Lowercase and remove accents so 'Móvil' matches 'movil'."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def col(df, name):
    """Return a text column, or empty text if the column is missing."""
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def whole_word(text, keywords):
    """First keyword found as a whole word (plural 's' allowed)."""
    text = clean(text)
    for word in keywords:
        w = re.escape(clean(word))
        if re.search(rf"\b{w}s?\b", text):
            return word
    return ""


def word_start(text, keywords):
    """First keyword found at the start of a word ('cloud' finds 'cloudflare')."""
    text = clean(text)
    for word in keywords:
        w = re.escape(clean(word))
        if re.search(rf"\b{w}", text):
            return word
    return ""


def is_priority(company, names):
    company = clean(company)
    for n in names:
        if re.search(rf"\b{re.escape(clean(n))}\b", company):
            return True
    return False


def join_unique(values, sep):
    seen = []
    for v in values:
        v = str(v).strip()
        if v and v not in seen:
            seen.append(v)
    return sep.join(seen)


def main():
    with open("criteria.yaml", encoding="utf-8") as f:
        c = yaml.safe_load(f)

    fetch_desc = c.get("linkedin_fetch_description", True)
    give_up_after = c.get("skip_site_after_failures", 4)

    sites = list(c.get("sites", [])) + list(c.get("optional_sites", []))
    tries = {s: 0 for s in sites}
    wins = {s: 0 for s in sites}
    found = {s: 0 for s in sites}
    skipped = set()
    log = []
    frames = []

    for loc in c["locations"]:
        for term in c["search_terms"]:
            for site in sites:
                if site in skipped:
                    continue
                if site == "zip_recruiter" and loc["country"] not in ("USA", "Canada"):
                    continue
                label = f"{site} | {term} | {loc['name']}"
                tries[site] += 1
                try:
                    kwargs = dict(
                        site_name=[site],
                        search_term=term,
                        google_search_term=f"{term} jobs near {loc['name']} since yesterday",
                        location=loc["name"],
                        results_wanted=c.get("results_per_search", 20),
                        hours_old=c.get("days_back", 7) * 24,
                        country_indeed=loc["country"],
                    )
                    if site == "linkedin" and fetch_desc:
                        kwargs["linkedin_fetch_description"] = True
                    df = scrape_jobs(**kwargs)
                    if df is None or df.empty:
                        log.append(f"EMPTY  {label}")
                    else:
                        df["searched_location"] = loc["name"]
                        frames.append(df)
                        wins[site] += 1
                        found[site] += len(df)
                        log.append(f"OK     {label} -> {len(df)} jobs")
                except Exception as err:
                    log.append(f"FAILED {label} -> {str(err)[:120]}")
                if wins[site] == 0 and tries[site] >= give_up_after:
                    skipped.add(site)
                    log.append(f"SKIPPED {site}: no jobs after {tries[site]} tries")
                time.sleep(random.uniform(3, 7))

    os.makedirs("results", exist_ok=True)

    log.append("")
    log.append("=== SITE SUMMARY ===")
    for s in sites:
        state = "skipped" if s in skipped else "used"
        log.append(f"{s}: {found[s]} jobs from {wins[s]}/{tries[s]} searches ({state})")

    if not frames:
        log.append("No jobs found from any site.")
        with open("results/run_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        print("\n".join(log))
        return

    jobs = pd.concat(frames, ignore_index=True)
    jobs = jobs.drop_duplicates(subset="job_url").reset_index(drop=True)
    log.append("")
    log.append("=== FILTERS ===")
    log.append(f"Total after removing duplicates: {len(jobs)}")

    title = col(jobs, "title")
    desc = col(jobs, "description")
    company = col(jobs, "company")
    industry_col = col(jobs, "company_industry")
    level_col = col(jobs, "job_level")
    type_col = col(jobs, "job_type")

    # 1. Seniority: the title must contain a seniority word
    sen_words = c.get("seniority_keywords", [])
    sen_found = title.apply(lambda t: whole_word(t, sen_words))
    keep = sen_found != ""
    seniority = sen_found.where(level_col == "", level_col)
    log.append(f"Dropped for seniority: {int((~keep).sum())}")

    # 2. Employment type: keep a match, or when the site gave no type
    types = [clean(t) for t in c.get("employment_types", [])]
    if types:
        type_keep = type_col.apply(
            lambda v: clean(v) == "" or any(t in clean(v) for t in types)
        )
        log.append(f"Dropped for employment type: {int((keep & ~type_keep).sum())}")
        keep = keep & type_keep

    # 3. Exclude words in the title
    ex_words = c.get("exclude_title_words", [])
    excluded = title.apply(lambda t: whole_word(t, ex_words) != "")
    log.append(f"Dropped for exclude words: {int((keep & excluded).sum())}")
    keep = keep & ~excluded

    # 4. Relevance score
    pri_names = c.get("priority_companies", [])
    is_pri = company.apply(lambda v: is_priority(v, pri_names))

    ind_words = c.get("industries", [])
    blob = title + " " + company + " " + desc + " " + industry_col
    ind_found = blob.apply(lambda t: word_start(t, ind_words))
    ind_ok = (ind_found != "") | is_pri
    has_info = (desc != "") | (industry_col != "")

    title_hit = title.apply(lambda t: whole_word(t, c.get("title_keywords", [])) != "")
    func_title = title.apply(lambda t: whole_word(t, c.get("job_functions", [])))

    score = (
        title_hit.astype(int) * 3
        + ind_ok.astype(int) * 2
        + (func_title != "").astype(int) * 1
    )
    min_score = c.get("min_relevance_score", 3)
    low = score < min_score
    log.append(f"Dropped for low relevance score (< {min_score}): {int((keep & low).sum())}")
    keep = keep & ~low

    industry_status = pd.Series("Unverified", index=jobs.index)
    industry_status[ind_found != ""] = "Verified"
    industry_status[(ind_found == "") & is_pri] = "Priority company"
    industry_status[(ind_found == "") & ~is_pri & has_info] = "No match in text"
    industry = industry_col.where(industry_col != "", ind_found)

    func = func_title.where(
        func_title != "",
        (title + " " + desc).apply(lambda t: whole_word(t, c.get("job_functions", []))),
    )
    func = func.where(func != "", "Other")

    rank_score = score + is_pri.astype(int)

    out = pd.DataFrame(
        {
            "Score": rank_score,
            "Priority": is_pri.map({True: "Yes", False: ""}),
            "Title": title,
            "Company": company,
            "Location": col(jobs, "location"),
            "Employment type": type_col,
            "Industry": industry,
            "Industry check": industry_status,
            "Job function": func,
            "Seniority level": seniority,
            "Site": col(jobs, "site"),
            "Date posted": col(jobs, "date_posted"),
            "Job link": col(jobs, "job_url"),
        }
    )[keep]

    # Sort by score, then newest
    out["_date"] = pd.to_datetime(out["Date posted"], errors="coerce")
    out = out.sort_values(["Score", "_date"], ascending=[False, False])
    out["_key"] = out["Title"].apply(clean) + "|" + out["Company"].apply(clean)

    # 5. Merge repeats (same title + company) into one row
    before = len(out)
    merged = out.groupby("_key", sort=False).agg(
        {
            "Score": "max",
            "Priority": "first",
            "Title": "first",
            "Company": "first",
            "Location": lambda s: join_unique(s, " | "),
            "Employment type": "first",
            "Industry": "first",
            "Industry check": "first",
            "Job function": "first",
            "Seniority level": "first",
            "Site": lambda s: join_unique(s, ", "),
            "Date posted": "first",
            "Job link": "first",
            "_date": "first",
        }
    )
    others = out.groupby("_key", sort=False)["Job link"].apply(
        lambda s: " ".join(list(s)[1:])
    )
    merged["Other links"] = others
    out = merged.reset_index(drop=True)
    out = out.sort_values(["Score", "_date"], ascending=[False, False])
    out = out.drop(columns=["_date"]).reset_index(drop=True)
    log.append(f"Merged repeated jobs: {before - len(out)}")

    log.append(f"Final jobs kept: {len(out)}")
    log.append(f"  Priority companies: {int((out['Priority'] == 'Yes').sum())}")
    log.append(f"  Industry verified: {int((out['Industry check'] == 'Verified').sum())}")
    log.append(f"  Score 5 or more: {int((out['Score'] >= 5).sum())}")

    today = date.today().isoformat()
    out.to_csv(f"results/jobs_{today}.csv", index=False, encoding="utf-8-sig")
    out.to_csv("results/latest.csv", index=False, encoding="utf-8-sig")
    with open("results/run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("\n".join(log))


if __name__ == "__main__":
    main()
