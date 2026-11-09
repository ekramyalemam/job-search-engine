"""Phase 1 (v2): search job boards using the settings in criteria.yaml."""
import os
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
    """First keyword found at the start of a word ('network' finds 'networking')."""
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
                time.sleep(2)

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

    # Seniority: the title must contain a seniority word (whole words only)
    sen_words = c.get("seniority_keywords", [])
    sen_found = title.apply(lambda t: whole_word(t, sen_words))
    keep = sen_found != ""
    seniority = sen_found.where(level_col == "", level_col)
    log.append(f"Dropped for seniority: {int((~keep).sum())}")

    # Industry: Verified / Unverified / dropped
    ind_words = c.get("industries", [])
    blob = title + " " + desc + " " + industry_col
    ind_found = blob.apply(lambda t: word_start(t, ind_words))
    has_info = (desc != "") | (industry_col != "")
    verified = ind_found != ""
    ind_keep = verified | ~has_info
    industry_status = pd.Series("Unverified", index=jobs.index)
    industry_status[verified] = "Verified"
    industry = industry_col.where(industry_col != "", ind_found)
    log.append(f"Dropped for industry: {int((keep & ~ind_keep).sum())}")
    keep = keep & ind_keep

    # Employment type: keep a match, or when the site gave no type
    types = [clean(t) for t in c.get("employment_types", [])]
    if types:
        type_keep = type_col.apply(
            lambda v: clean(v) == "" or any(t in clean(v) for t in types)
        )
        log.append(f"Dropped for employment type: {int((keep & ~type_keep).sum())}")
        keep = keep & type_keep

    # Job function label
    func = (title + " " + desc).apply(lambda t: whole_word(t, c.get("job_functions", [])))
    func = func.where(func != "", "Other")

    # Priority companies (a bonus, never a filter)
    pri_names = c.get("priority_companies", [])
    priority = company.apply(lambda v: "Yes" if is_priority(v, pri_names) else "")

    out = pd.DataFrame(
        {
            "Priority": priority,
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

    # Sort: priority first, then Verified industry, then newest
    out["_date"] = pd.to_datetime(out["Date posted"], errors="coerce")
    out["_p"] = (out["Priority"] != "Yes").astype(int)
    out["_v"] = (out["Industry check"] != "Verified").astype(int)
    out = out.sort_values(["_p", "_v", "_date"], ascending=[True, True, False])
    out = out.drop(columns=["_date", "_p", "_v"]).reset_index(drop=True)

    log.append(f"Final jobs kept: {len(out)}")
    log.append(f"  Priority companies: {int((out['Priority'] == 'Yes').sum())}")
    log.append(f"  Industry verified: {int((out['Industry check'] == 'Verified').sum())}")
    log.append(f"  Industry unverified: {int((out['Industry check'] == 'Unverified').sum())}")

    today = date.today().isoformat()
    out.to_csv(f"results/jobs_{today}.csv", index=False, encoding="utf-8-sig")
    out.to_csv("results/latest.csv", index=False, encoding="utf-8-sig")
    with open("results/run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("\n".join(log))


if __name__ == "__main__":
    main()
