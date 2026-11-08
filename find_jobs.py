"""Phase 1: search job boards using the settings in criteria.yaml."""
import os
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


def first_match(text, keywords):
    """Return the first keyword found in the text, or empty text."""
    text = clean(text)
    for word in keywords:
        if clean(word) in text:
            return word
    return ""


def main():
    with open("criteria.yaml", encoding="utf-8") as f:
        c = yaml.safe_load(f)

    sites = list(c.get("sites", [])) + list(c.get("optional_sites", []))
    log = []
    frames = []

    for loc in c["locations"]:
        for term in c["search_terms"]:
            for site in sites:
                if site == "zip_recruiter" and loc["country"] not in ("USA", "Canada"):
                    continue
                label = f"{site} | {term} | {loc['name']}"
                try:
                    df = scrape_jobs(
                        site_name=[site],
                        search_term=term,
                        google_search_term=f"{term} jobs near {loc['name']}",
                        location=loc["name"],
                        results_wanted=c.get("results_per_search", 20),
                        hours_old=c.get("days_back", 7) * 24,
                        country_indeed=loc["country"],
                    )
                    if df is None or df.empty:
                        log.append(f"EMPTY  {label}")
                    else:
                        df["searched_location"] = loc["name"]
                        frames.append(df)
                        log.append(f"OK     {label} -> {len(df)} jobs")
                except Exception as err:
                    log.append(f"FAILED {label} -> {str(err)[:120]}")
                time.sleep(2)

    os.makedirs("results", exist_ok=True)

    if not frames:
        log.append("No jobs found from any site.")
        with open("results/run_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        print("\n".join(log))
        return

    jobs = pd.concat(frames, ignore_index=True)
    jobs = jobs.drop_duplicates(subset="job_url")
    log.append(f"Total after removing duplicates: {len(jobs)}")

    title = col(jobs, "title")
    desc = col(jobs, "description")
    company = col(jobs, "company")
    industry_col = col(jobs, "company_industry")
    level_col = col(jobs, "job_level")
    type_col = col(jobs, "job_type")

    # Seniority: keep jobs whose title has a seniority word
    seniority = title.apply(lambda t: first_match(t, c.get("seniority_keywords", [])))
    seniority = seniority.where(level_col == "", level_col)
    keep = title.apply(lambda t: first_match(t, c.get("seniority_keywords", [])) != "")
    log.append(f"Dropped for seniority: {int((~keep).sum())}")

    # Industry: keep on a match, or when we have no industry info to check
    ind_words = c.get("industries", [])
    blob = title + " " + desc + " " + industry_col
    industry = blob.apply(lambda t: first_match(t, ind_words))
    industry = industry.where(industry_col == "", industry_col)
    no_info = (desc == "") & (industry_col == "")
    ind_keep = (blob.apply(lambda t: first_match(t, ind_words) != "")) | no_info
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

    # Optional: only your chosen companies
    if c.get("only_these_companies"):
        names = [clean(n) for n in c.get("companies", [])]
        co_keep = company.apply(lambda v: any(n in clean(v) for n in names))
        log.append(f"Dropped for company list: {int((keep & ~co_keep).sum())}")
        keep = keep & co_keep

    # Job function label
    func = (title + " " + desc).apply(lambda t: first_match(t, c.get("job_functions", [])))
    func = func.where(func != "", "Other")

    out = pd.DataFrame(
        {
            "Title": title,
            "Company": company,
            "Location": col(jobs, "location"),
            "Employment type": type_col,
            "Industry": industry,
            "Job function": func,
            "Seniority level": seniority,
            "Site": col(jobs, "site"),
            "Date posted": col(jobs, "date_posted"),
            "Job link": col(jobs, "job_url"),
        }
    )[keep]

    out = out.reset_index(drop=True)
    log.append(f"Final jobs kept: {len(out)}")

    today = date.today().isoformat()
    out.to_csv(f"results/jobs_{today}.csv", index=False, encoding="utf-8-sig")
    out.to_csv("results/latest.csv", index=False, encoding="utf-8-sig")
    with open("results/run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("\n".join(log))


if __name__ == "__main__":
    main()