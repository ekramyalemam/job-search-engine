"""
=========================================================
find_jobs.py - v12 (Hybrid + bonus-only companies + no-engineer filter)
Features: Remote/Hybrid detection, Executive Gates, +3 Priority Boost
=========================================================
"""
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
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()

def col(df, name):
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)

def whole_word(text, keywords):
    text = clean(text)
    for word in keywords:
        w = re.escape(clean(word))
        if re.search(rf"\b{w}s?\b", text):
            return word
    return ""

def word_start(text, keywords):
    text = clean(text)
    for word in keywords:
        w = re.escape(clean(word))
        if re.search(rf"\b{w}", text):
            return word
    return ""

def is_priority(company, names, not_names):
    company = clean(company)
    for bad in not_names:
        if clean(bad) in company:
            return False
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

def region(location):
    s = clean(location)
    for key, name in [
        ("united arab", "UAE"), ("uae", "UAE"), ("dubai", "UAE"), ("abu dhabi", "UAE"),
        ("qatar", "Qatar"), ("doha", "Qatar"),
        ("saudi", "KSA"), ("riyadh", "KSA"), ("jeddah", "KSA"),
        ("canada", "Canada"), ("ontario", "Canada"),
        ("united states", "USA"), ("usa", "USA"), ("texas", "USA"),
    ]:
        if key in s:
            return name
    return "Other"

def tidy(d):
    d = d.drop(columns=["_tier"]).sort_values(["Score", "_date"], ascending=[False, False])
    if d.empty:
        d = d.drop(columns=["_key", "_date"])
        d["Other links"] = ""
        return d.reset_index(drop=True), 0
    grp = d.groupby("_key", sort=False)
    merged = grp.agg({
        "Score": "max",
        "Priority": "first",
        "Title": "first",
        "Company": "first",
        "Workplace": "first",
        "Location": lambda s: join_unique(s, " | "),
        "Employment type": "first",
        "Industry": "first",
        "Industry check": "first",
        "Job function": "first",
        "Seniority level": "first",
        "Site": lambda s: join_unique(s, ", "),
        "Date posted": "first",
        "Job link": "first",
        "Reason": "first",
        "_date": "first",
    })
    merged["Other links"] = grp["Job link"].apply(lambda s: " ".join(list(s)[1:]))
    merged = merged.reset_index(drop=True)
    merged = merged.sort_values(["Score", "_date"], ascending=[False, False])
    merged = merged.drop(columns=["_date"]).reset_index(drop=True)
    return merged, len(d) - len(merged)

def short(text, n):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"

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
    today = date.today().isoformat()

    site_lines = []
    for s in sites:
        state = "skipped" if s in skipped else "used"
        site_lines.append(f"{s}: {found[s]} jobs from {wins[s]}/{tries[s]} searches ({state})")
    log.append("")
    log.append("=== SITE SUMMARY ===")
    log.extend(site_lines)

    if not frames:
        log.append("No jobs found from any site.")
        with open("results/run_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        print("\n".join(log))
        return

    jobs = pd.concat(frames, ignore_index=True)
    jobs = jobs.drop_duplicates(subset="job_url").reset_index(drop=True)
    total = len(jobs)

    title = col(jobs, "title")
    desc = col(jobs, "description")
    company = col(jobs, "company")
    industry_col = col(jobs, "company_industry")
    level_col = col(jobs, "job_level")
    type_col = col(jobs, "job_type")
    idx = jobs.index

    # Workplace Parsing (Remote / Hybrid / Onsite)
    is_rem = col(jobs, "is_remote").str.lower()
    loc_lower = col(jobs, "location").str.lower()
    desc_lower = desc.str.lower()
    
    workplace = pd.Series("Onsite", index=idx)
    workplace = workplace.mask(desc_lower.str.contains(r"\bhybrid\b", regex=True) | loc_lower.str.contains(r"\bhybrid\b", regex=True), "Hybrid")
    workplace = workplace.mask((is_rem == "true") | loc_lower.str.contains(r"\bremote\b", regex=True) | desc_lower.str.contains(r"\bremote\b", regex=True), "Remote")

    # Hard rules: Role Gate, Seniority, Employment Type, Exclude Words
    role_found = title.apply(lambda t: word_start(t, c.get("role_keywords", [])))
    sen_found = title.apply(lambda t: whole_word(t, c.get("seniority_keywords", [])))
    seniority = sen_found.where(level_col == "", level_col)

    types = [clean(t) for t in c.get("employment_types", [])]
    if types:
        type_keep = type_col.apply(lambda v: clean(v) == "" or any(t in clean(v) for t in types))
    else:
        type_keep = pd.Series(True, index=idx)

    mgmt_found = title.apply(lambda t: whole_word(t, c.get("management_words", [])))
    ic_hit = title.apply(lambda t: whole_word(t, c.get("individual_contributor_words", [])))

    ex_hit = title.apply(lambda t: whole_word(t, c.get("exclude_title_words", [])))
    title_hit = title.apply(lambda t: whole_word(t, c.get("title_keywords", [])) != "")

    # Relevance score
    pri_names = c.get("priority_companies", [])
    not_pri = c.get("not_priority_companies", [])
    is_pri = company.apply(lambda v: is_priority(v, pri_names, not_pri))

    blob = title + " " + company + " " + desc + " " + industry_col
    ind_found = blob.apply(lambda t: word_start(t, c.get("industries", [])))
    bonus_only = c.get("bonus_only_companies", []) or []
    is_broad = company.apply(lambda v: is_priority(v, bonus_only, not_pri))
    ind_ok = (ind_found != "") | (is_pri & ~is_broad)
    has_info = (desc != "") | (industry_col != "")

    func_title = title.apply(lambda t: whole_word(t, c.get("job_functions", [])))

    adj_words = c.get("adjacent_title_keywords", []) or []
    adj_hit = title.apply(lambda t: whole_word(t, adj_words) != "") & ~title_hit

    score = (
        title_hit.astype(int) * 3
        + ind_ok.astype(int) * 2
        + (func_title != "").astype(int)
        + adj_hit.astype(int)
    )
    min_score = c.get("min_relevance_score", 3)

    reason = pd.Series("", index=idx)
    reason = reason.mask((reason == "") & (role_found == ""), "missing role keyword")
    reason = reason.mask((reason == "") & (sen_found == ""), "no seniority word in title")
    reason = reason.mask((reason == "") & (ic_hit != "") & (mgmt_found == ""), "engineer/specialist title: " + ic_hit)
    reason = reason.mask((reason == "") & ~type_keep, "employment type: " + type_col)
    reason = reason.mask((reason == "") & (ex_hit != "") & ~title_hit, "exclude word: " + ex_hit)
    
    passed = reason == ""
    rescued = passed & (ex_hit != "")
    
    tier = pd.Series("rejected", index=idx)
    tier[passed & (score >= min_score)] = "main"
    tier[passed & (score < min_score) & (score >= 1)] = "maybe"
    tier[rescued] = "maybe"
    reason[tier == "maybe"] = "score " + score.astype(str) + " below " + str(min_score)
    reason[rescued] = "exclude word '" + ex_hit + "' but title has a tech word"
    reason[tier == "main"] = ""
    reason[passed & (tier == "rejected")] = "score 0: no role or industry evidence"

    industry_status = pd.Series("Unverified", index=idx)
    industry_status[ind_found != ""] = "Verified"
    industry_status[(ind_found == "") & is_pri] = "Priority company"
    industry_status[(ind_found == "") & ~is_pri & has_info] = "No match in text"
    industry = industry_col.where(industry_col != "", ind_found)

    func = func_title.where(
        func_title != "",
        (title + " " + desc).apply(lambda t: whole_word(t, c.get("job_functions", []))),
    )
    func = func.where(func != "", "Other")

    pref = c.get("preferred_locations", []) or []
    loc_bonus = col(jobs, "location").apply(lambda v: 1 if word_start(v, pref) != "" else 0)

    # Apply Score, Priority Bonus (+3), and Location Bonus
    full = pd.DataFrame({
        "Score": score + (is_pri.astype(int) * 3) + loc_bonus,
        "Priority": is_pri.map({True: "Yes", False: ""}),
        "Title": title,
        "Company": company,
        "Workplace": workplace,
        "Location": col(jobs, "location"),
        "Employment type": type_col,
        "Industry": industry,
        "Industry check": industry_status,
        "Job function": func,
        "Seniority level": seniority,
        "Site": col(jobs, "site"),
        "Date posted": col(jobs, "date_posted"),
        "Job link": col(jobs, "job_url"),
        "Reason": reason,
        "_tier": tier,
    })
    
    full["_date"] = pd.to_datetime(full["Date posted"], errors="coerce")
    full["_key"] = full["Title"].apply(clean) + "|" + full["Company"].apply(clean)

    main_df, merged_main = tidy(full[full["_tier"] == "main"])
    maybe_df, merged_maybe = tidy(full[full["_tier"] == "maybe"])
    main_df = main_df.drop(columns=["Reason"])
    cols = [x for x in maybe_df.columns if x not in ("Other links", "Reason")]
    maybe_df = maybe_df[cols + ["Other links", "Reason"]]

    rej = full[full["_tier"] == "rejected"].drop_duplicates("Job link")
    rej_df = rej[["Reason", "Title", "Company", "Workplace", "Location", "Site", "Job link"]]
    rej_df = rej_df.sort_values(["Reason", "Company"]).reset_index(drop=True)

    r = reason
    n_role = int(r.str.startswith("missing role keyword").sum())
    n_sen = int(r.str.startswith("no seniority").sum())
    n_emp = int(r.str.startswith("employment").sum())
    n_ic = int(r.str.startswith("engineer/specialist").sum())
    n_exc = int(r.str.startswith("exclude word:").sum())
    n_resc = int(rescued.sum())
    n_zero = int(r.str.startswith("score 0").sum())
    filt = [
        f"Total after removing duplicates: {total}",
        f"Dropped for missing role keyword: {n_role}",
        f"Dropped for seniority: {n_sen}",
        f"Dropped for employment type: {n_emp}",
        f"Dropped as engineer/specialist (not a manager): {n_ic}",
        f"Dropped for exclude words: {n_exc}",
        f"Rescued to MAYBE (exclude word but tech title): {n_resc}",
        f"Dropped, score 0 (no evidence): {n_zero}",
        f"Moved to MAYBE list (score 1-{min_score - 1}): {int((tier == 'maybe').sum())}",
        f"Strong matches before merging repeats: {int((tier == 'main').sum())}",
        f"Merged repeated jobs: {merged_main}",
        f"Final strong matches: {len(main_df)}",
        f"Final maybe list: {len(maybe_df)}",
        f"  Priority companies: {int((main_df['Priority'] == 'Yes').sum())}",
        f"  Industry verified: {int((main_df['Industry check'] == 'Verified').sum())}",
        f"  Score 5 or more: {int((main_df['Score'] >= 5).sum())}",
    ]
    log.append("")
    log.append("=== FILTERS ===")
    log.extend(filt)

    main_df.to_csv(f"results/jobs_{today}.csv", index=False, encoding="utf-8-sig")
    main_df.to_csv("results/latest.csv", index=False, encoding="utf-8-sig")
    maybe_df.to_csv("results/maybe_latest.csv", index=False, encoding="utf-8-sig")
    rej_df.to_csv("results/rejected_latest.csv", index=False, encoding="utf-8-sig")
    # NEW (for Phase 2): save the job text, matched to each job by its link. Nothing else changes.
    pd.DataFrame({"Job link": col(jobs, "job_url"), "Description": desc.str.slice(0, 8000)}).to_csv(
        "results/descriptions_latest.csv", index=False, encoding="utf-8-sig")
    with open("results/run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    reg = main_df["Location"].apply(region).value_counts().to_dict()
    sc = main_df["Score"].value_counts().sort_index(ascending=False).to_dict()
    lines = [f"RUN SUMMARY {today}", "", "SITES"]
    lines += site_lines
    lines += ["", "FILTERS"] + filt
    lines += ["", "REGIONS (strong matches): " + ", ".join(f"{k} {v}" for k, v in reg.items())]
    lines += ["SCORES: " + ", ".join(f"{k}:{v}" for k, v in sc.items())]
    lines += ["", "TOP 15 (score | title | workplace | company | location)"]
    for _, row in main_df.head(15).iterrows():
        lines.append(f"{row['Score']} | {short(row['Title'], 40)} | {row['Workplace']} | {short(row['Company'], 20)} | {short(row['Location'], 20)}")
    lines += ["", "LOWEST 8 STRONG MATCHES"]
    for _, row in main_df.tail(8).iterrows():
        lines.append(f"{row['Score']} | {short(row['Title'], 40)} | {row['Workplace']} | {short(row['Company'], 20)} | {row['Industry check']}")
    lines += ["", "TOP 10 MAYBE (score | title | workplace | company | why)"]
    for _, row in maybe_df.head(10).iterrows():
        lines.append(f"{row['Score']} | {short(row['Title'], 40)} | {row['Workplace']} | {short(row['Company'], 20)} | {row['Reason']}")
    lines += ["", "DROPPED SAMPLES (to check nothing good is lost)"]
    for label, key in [
        ("missing role", "missing role keyword"),
        ("seniority", "no seniority"),
        ("engineer/specialist", "engineer/specialist"),
        ("exclude word", "exclude word:"),
        ("score 0", "score 0"),
    ]:
        sub = rej_df[rej_df["Reason"].str.startswith(key)]
        lines.append(f"- {label}: {len(sub)} total; examples:")
        for _, row in sub.sample(min(6, len(sub)), random_state=1).iterrows():
            lines.append(f"    {short(row['Title'], 40)} | {row['Workplace']} | {short(row['Company'], 20)} | {short(row['Reason'], 20)}")
    text = "\n".join(lines)
    with open("results/share_me.txt", "w", encoding="utf-8") as f:
        f.write(text)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")

    print("\n".join(log))

if __name__ == "__main__":
    main()
