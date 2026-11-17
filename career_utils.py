"""
career_utils.py - shared helpers so historical (2025 snapshot) and live (Adzuna) data go through
the SAME steps:  text cleaning -> title rules -> family label -> text-based skill extraction.

Usage (Colab):
    import sys; sys.path.append("/content/drive/MyDrive/Job recommendation nd skill gap analysis")
    from career_utils import *
"""
import re, html, json, pickle
from pathlib import Path
import pandas as pd

OTHER = "Other / Non-IT or Unclear"
AMBIG = "IT Services - Ambiguous Role"

# ----------------------------------------------------------------- cleaning (same as notebook)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HTML_RE = re.compile(r"<[^>]+>")
CITY_FIX = {"bangalore": "Bengaluru", "gurgaon": "Gurugram", "bombay": "Mumbai",
            "madras": "Chennai", "calcutta": "Kolkata", "new delhi": "Delhi"}

def clean_text(value):
    if value is None or (isinstance(value, float) and value != value):
        return ""
    s = html.unescape(str(value))
    for rx in (HTML_RE, URL_RE, EMAIL_RE, PHONE_RE):
        s = rx.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_text(value):
    s = clean_text(value).lower()
    s = re.sub(r"[^a-z0-9+#./ -]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_city(value):
    if not value:
        return "Unknown"
    first = re.split(r"[,(/]", str(value))[0].strip()
    return CITY_FIX.get(first.lower(), first.title()) if first else "Unknown"

# ----------------------------------------------------------------- title rules (final version v3)
FAMILY_RULES = [
    ("Cybersecurity", r"\bcyber ?security\b|\binformation security\b|\binfosec\b|\bsecurity analyst\b|\bsecurity engineer\b|penetration test|\bvapt\b|\bsoc analyst\b|\biam engineer\b"),
    ("Data Scientist / ML-AI", r"data scientist|machine learning|\bartificial intelligence\b|\bai engineer\b|\bml engineer\b|deep learning|computer vision|\bnlp engineer\b|generative ai|\bllm engineer\b|\bdata science\b"),
    ("Data Engineer", r"data engineer|data engineering|etl developer|etl engineer|data pipeline|data platform engineer|big data engineer|data warehouse engineer"),
    ("Data / BI Analyst", r"data analyst|business intelligence|\bbi analyst\b|power bi|tableau developer|reporting analyst|analytics analyst|business data analyst"),
    ("Database Admin", r"database administrator|\bdba\b|database engineer|database developer|oracle database admin|sql database administrator"),
    ("DevOps / Cloud", r"devops|site reliability|\bsre\b|cloud engineer|cloud architect|cloud administrator|platform engineer|kubernetes engineer|\baws engineer\b|\bazure engineer\b|\bgcp engineer\b|infrastructure engineer"),
    ("QA / Testing", r"quality assurance|\bqa engineer\b|\bqa analyst\b|software tester|test engineer|testing engineer|automation test|\bsdet\b"),
    ("Mobile Developer", r"android developer|ios developer|mobile app developer|mobile developer|flutter developer|react native developer|swift developer|kotlin developer"),
    ("Frontend / Web", r"frontend|front end developer|web developer|ui developer|react developer|angular developer|vue developer|wordpress developer|php developer"),
    ("ERP / CRM Consultant", r"\bsap\b|salesforce|servicenow|dynamics 365|workday consultant|oracle (fusion|apps|erp)|\berp consultant\b|\bcrm consultant\b|peoplesoft"),
    ("Business / Systems Analyst", r"business analyst|systems analyst|system analyst|functional analyst|technical business analyst|business systems analyst"),
    ("IT Support / Sys & Network", r"it support|technical support engineer|help ?desk|desktop support|system administrator|systems administrator|network engineer|network administrator|service desk|application support|production support"),
    ("Tech Project / Product", r"technical project manager|it project manager|technical program manager|scrum master|technical product manager|product owner|delivery manager"),
    ("Software Developer", r"software engineer|software developer|application developer|full ?stack developer|backend developer|back end developer|java developer|python developer|c\+\+ developer|golang developer|node\.js developer|\.net developer|programmer|\bsde[ -]?[123]?\b|embedded software"),
]
EXTRA_RULES = [
    ("Cybersecurity", r"security architect"),
    ("Data Engineer", r"data architect|data solution architect|snowflake developer"),
    ("QA / Testing", r"test automation engineer|quality engineer tester|test lead|quality engineering lead"),
    ("Frontend / Web", r"react ?js|angular|vue ?js|front ?end"),
    ("Software Developer", r"software development (engineer|lead)|\bsde\b|dot ?net|full ?stack|java lead|mainframe developer|mulesoft developer|node ?js developer|team leader\s*-?\s*software|developer\s*-?\s*l\d|software (engineer|developer|programmer)"),
    ("IT Architect", r"(solution|technology|technical|enterprise) architect"),
    (AMBIG, r"^technical lead|application (lead|designer|tech support)"),
]
EXTRA_RULES_2 = [
    ("Software Developer", r"\.net|dot ?net|\bmern\b|\bmean stack\b|sql developer|as400|java architect|shopify developer|senior engineer ?-? ?software|software (engineer|developer)"),
    ("Data Engineer", r"big data developer|databricks|\betl\b"),
    ("Data Scientist / ML-AI", r"large language model|\bllm\b|generative ai|gen ?ai"),
    ("QA / Testing", r"qa automation|test engineer|\bsdet\b|quality assurance"),
    ("IT Support / Sys & Network", r"technology support|system(s)? administrator|sas administrator|helpdesk"),
    ("Cybersecurity", r"offensive security"),
]
NEGATIVE = (r"\bcnc\b|\bvmc\b|\bplc\b|machine operator|\bautocad\b|\bmechanical\b|\bcivil\b|"
            r"\belectrical\b|\binstrumentation\b|recruiter|talent acquisition|"
            r"business development|\bsales\b|\bteacher\b")

def assign_family(title):
    t = normalize_text(title)
    fam = OTHER
    for family, pat in FAMILY_RULES:
        if re.search(pat, t, flags=re.I):
            fam = family; break
    if fam == OTHER:
        for family, pat in EXTRA_RULES:
            if re.search(pat, t, flags=re.I):
                fam = family; break
    if re.search(NEGATIVE, t):
        return OTHER
    if fam == OTHER:
        for family, pat in EXTRA_RULES_2:
            if re.search(pat, t, flags=re.I):
                fam = family; break
    if fam == "Frontend / Web" and re.search(r"full ?stack", t):
        fam = "Software Developer"
    return fam

# ----------------------------------------------------------------- text-based skill extraction
SKILL_ALIASES = {"aws": "amazon web services", "gcp": "google cloud platform", "k8s": "kubernetes",
                 "ci/cd": "continuous integration/continuous delivery", "reactjs": "react.js", "react": "react.js",
                 "nodejs": "node.js", "node": "node.js", "dotnet": ".net", "ml": "machine learning",
                 "nlp": "natural language processing", "js": "javascript", "powerbi": "power bi"}

def _tok(s):
    return [t.strip(".") or t for t in re.findall(r"[a-z0-9#+.]+", s.lower())]

class SkillExtractor:
    """Dictionary (longest-match n-gram) extractor over the 1,951-skill vocabulary."""
    def __init__(self, vocab):
        self.vocab = set(vocab)
        self.table = {tuple(_tok(s)): s for s in self.vocab}
        for a, c in SKILL_ALIASES.items():
            if c in self.vocab:
                self.table[tuple(_tok(a))] = c
        self.maxn = max(len(k) for k in self.table)

    def extract(self, text):
        t, found, i = _tok(text), [], 0
        while i < len(t):
            hit = None
            for n in range(min(self.maxn, len(t) - i), 0, -1):
                k = tuple(t[i:i + n])
                if k in self.table:
                    hit = (self.table[k], n); break
            if hit:
                found.append(hit[0]); i += hit[1]
            else:
                i += 1
        return list(dict.fromkeys(found))

def load_vocab(path):
    return pickle.load(open(path, "rb"))

# ----------------------------------------------------------------- Adzuna (live) -> same schema
def load_adzuna(raw_dir, extractor):
    rows = []
    for f in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    g = lambda col, key: a[col].map(lambda d: d.get(key) if isinstance(d, dict) else None)
    out = pd.DataFrame({
        "jobId": a["id"].astype(str),
        "source": "adzuna_live", "period": "live",
        "title": a["title"].fillna(""),
        "companyName": g("company", "display_name"),
        "location": g("location", "display_name"),
        "category": g("category", "label"),
        "created": pd.to_datetime(a["created"], errors="coerce", utc=True),
        "salary_min_lpa": a.get("salary_min") / 1e5 if "salary_min" in a else None,
        "salary_max_lpa": a.get("salary_max") / 1e5 if "salary_max" in a else None,
        "description_clean": a["description"].map(clean_text),
        "query": a.get("query"), "collected_on": a.get("collected_on"),
    })
    out["title_clean"] = out["title"].map(normalize_text)
    out["city"] = out["location"].map(normalize_city)
    out["family"] = out["title"].map(assign_family)
    out["skills_txt"] = out["description_clean"].map(extractor.extract)   # description only (no title -> no leakage)
    out["n_skills_txt"] = out["skills_txt"].str.len()
    out["desc_words"] = out["description_clean"].str.split().str.len()
    return out.drop_duplicates(["title_clean", "companyName", "location", "description_clean"]).reset_index(drop=True)
