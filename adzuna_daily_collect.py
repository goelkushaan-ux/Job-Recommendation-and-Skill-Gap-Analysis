"""
Daily Adzuna India collector (run once per day, e.g. in Colab).

Setup (Colab):
    1. Colab left sidebar -> Secrets: add ADZUNA_APP_ID and ADZUNA_APP_KEY, enable notebook access.
    2. In a cell:  from google.colab import drive, userdata ; drive.mount('/content/drive')
                   import os ; os.environ['ADZUNA_APP_ID']=userdata.get('ADZUNA_APP_ID') ; os.environ['ADZUNA_APP_KEY']=userdata.get('ADZUNA_APP_KEY')
                   %run /content/drive/MyDrive/<your folder>/adzuna_daily_collect.py
Limits used (Adzuna terms, checked Nov 2026): 25/min, 250/day, 1000/week, 2500/month.
This script stays well below them. Verify your own key's limits on the Adzuna dashboard.
"""
import os, json, time, datetime as dt
from pathlib import Path
import requests

BASE = Path(os.environ.get("CAREER_BASE", "/content/drive/MyDrive/Job recommendation nd skill gap analysis")) / "adzuna"
APP_ID, APP_KEY = os.environ.get("ADZUNA_APP_ID"), os.environ.get("ADZUNA_APP_KEY")
QUERIES = ["software developer", "java developer", "python developer", "frontend developer react",
           "data engineer", "data scientist", "data analyst", "devops engineer", "qa automation tester",
           "cyber security analyst", "sap consultant", "business analyst", "android developer",
           "database administrator", "cloud engineer", "technical project manager"]
PAGES = 3                       # 16 queries x 3 pages = 48 calls/day (50 results per call)
SAFE = {"day": 200, "week": 800, "month": 2000}   # below the official 250 / 1000 / 2500
SLEEP = 3                       # seconds between calls -> max 20/min (limit is 25/min)

def main():
    assert APP_ID and APP_KEY, "Set ADZUNA_APP_ID and ADZUNA_APP_KEY first"
    (BASE / "raw").mkdir(parents=True, exist_ok=True)
    log_f, ids_f = BASE / "call_log.json", BASE / "seen_ids.json"
    log = json.loads(log_f.read_text()) if log_f.exists() else []          # list of ISO dates, one per call
    seen = set(json.loads(ids_f.read_text())) if ids_f.exists() else set()
    today = dt.date.today()

    def used(days):
        return sum(1 for d in log if (today - dt.date.fromisoformat(d)).days < days)

    out_f = BASE / "raw" / f"{today.isoformat()}.jsonl"
    new_total, calls, stop = 0, 0, False
    with out_f.open("a", encoding="utf-8") as out:
        for q in QUERIES:
            if stop:
                break
            for page in range(1, PAGES + 1):
                if used(1) >= SAFE["day"] or used(7) >= SAFE["week"] or used(30) >= SAFE["month"]:
                    print("Safe call budget reached - stopping."); stop = True; break
                r = requests.get(f"https://api.adzuna.com/v1/api/jobs/in/search/{page}",
                                 params={"app_id": APP_ID, "app_key": APP_KEY, "results_per_page": 50,
                                         "what": q, "sort_by": "date", "content-type": "application/json"},
                                 timeout=30)
                log.append(today.isoformat()); calls += 1
                if r.status_code != 200:
                    print(f"HTTP {r.status_code} for '{q}' p{page} - stopping.", r.text[:150]); stop = True; break
                results = r.json().get("results", [])
                for j in results:
                    jid = str(j.get("id"))
                    if jid in seen:
                        continue
                    seen.add(jid); new_total += 1
                    j["query"], j["collected_on"] = q, today.isoformat()
                    out.write(json.dumps(j, ensure_ascii=False) + "\n")
                if len(results) < 50:
                    break                       # is query ke aage pages nahi hain
                time.sleep(SLEEP)
    log_f.write_text(json.dumps(log)); ids_f.write_text(json.dumps(sorted(seen)))
    print(f"{today}: {calls} calls, {new_total} NEW postings, {len(seen)} unique so far -> {out_f.name}")
    print(f"Budget used: day {used(1)}/{SAFE['day']}, week {used(7)}/{SAFE['week']}, month {used(30)}/{SAFE['month']}")

if __name__ == "__main__":
    main()
