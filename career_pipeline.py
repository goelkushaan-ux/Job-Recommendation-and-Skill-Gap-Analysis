"""
career_pipeline.py - the data pipeline of the project, in one place, in a fixed order.

    load_and_clean   raw Excel -> cleaned, de-duplicated listings (93,729 rows)
    add_family       title rules -> family (career_utils.assign_family)
    add_skills       tag cleaning -> canonical skills + IT skill vocabulary (1,951 skills)
    build_detector   skills-only IT-vs-Other detector, out-of-fold scores (company-wise 5-fold)
    load_frozen_detector  the saved scores that the human validation was stratified on (used by default)
    build_extended   core / extended tiers + predicted families with prior correction (alpha = 0.5)

Every step is deterministic, so re-running gives the same numbers.
"""
import re, html
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from career_utils import OTHER, AMBIG, assign_family, HTML_RE, URL_RE, EMAIL_RE, PHONE_RE, CITY_FIX

ALPHA = 0.5            # prior correction for predicted families (frozen after 3 independent checks)
IT_CUTOFF = 0.15       # detector score above which an 'Other' listing joins the extended tier
HIGH_CUTOFF = 0.5

# ------------------------------------------------------------------ 1) load + clean
def _clean_series(series):
    s = series.fillna("").astype(str).map(html.unescape)
    for rx in (HTML_RE, URL_RE, EMAIL_RE, PHONE_RE):
        s = s.str.replace(rx, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def _parse_tags(value):
    if pd.isna(value):
        return []
    out, seen = [], set()
    for item in str(value).split(","):
        skill = re.sub(r"\s+", " ", item.lower()).strip(" .;:/")
        if skill and skill not in seen:
            seen.add(skill); out.append(skill)
    return out

def _city(value):
    if pd.isna(value):
        return "Unknown"
    first = re.split(r"[,(/]", str(value))[0].strip()
    return CITY_FIX.get(first.lower(), first.title()) if first else "Unknown"

def load_and_clean(raw_path):
    df = pd.read_excel(raw_path)
    need = ["title", "jobId", "companyName", "tagsAndSkills", "jobDescription", "location"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Dataset schema changed; missing columns: {miss}")
    n0 = len(df)
    df["jobId"] = df["jobId"].astype("string")
    df["title_clean"] = _clean_series(df["title"])
    df["title_norm"] = (df["title_clean"].str.lower().str.replace(r"[^a-z0-9+#./ -]", " ", regex=True)
                        .str.replace(r"\s+", " ", regex=True).str.strip())
    df["description_clean"] = _clean_series(df["jobDescription"])
    df["skills_list"] = df["tagsAndSkills"].map(_parse_tags)
    df["skills_text"] = df["skills_list"].map(" ".join)
    df["city"] = df["location"].map(_city)
    df["experience_min"] = pd.to_numeric(df.get("minimumExperience"), errors="coerce")
    df["experience_max"] = pd.to_numeric(df.get("maximumExperience"), errors="coerce")
    df = df.drop_duplicates(subset=["jobId"], keep="first")
    n1 = len(df)
    key = (df["title_norm"].fillna("") + "|" + df["companyName"].fillna("").astype(str).str.lower().str.strip() + "|"
           + df["location"].fillna("").astype(str).str.lower().str.strip() + "|"
           + df["description_clean"].fillna("").astype(str).str.lower()
             .str.replace(r"[^a-z0-9+#./ -]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
    df = df.loc[~key.duplicated(keep="first")]
    df["city"] = df["city"].str.replace(r"^Hybrid\s*-\s*", "", regex=True)
    print(f"rows: {n0:,} -> jobId dedupe {n1:,} -> content dedupe {len(df):,}")
    return df

# ------------------------------------------------------------------ 2) family
def add_family(df):
    df["family"] = df["title_clean"].map(assign_family)
    print("IT listings by title rules:", int(df["family"].ne(OTHER).sum()))
    return df

# ------------------------------------------------------------------ 3) skills
SKILL_ALIASES = {
    "machine learning": ["ml", "machine-learning"], "deep learning": ["dl", "deep-learning"],
    "natural language processing": ["nlp"], "computer vision": ["cv"], "generative ai": ["gen ai", "genai"],
    "large language models": ["llm", "llms", "large language model"], "scikit-learn": ["sklearn", "scikit learn"],
    "power bi": ["powerbi"], "microsoft excel": ["ms excel", "excel"], "amazon web services": ["aws"],
    "google cloud platform": ["gcp"], "continuous integration/continuous delivery": ["ci/cd", "cicd"],
    "javascript": ["js"], "typescript": ["ts"], "postgresql": ["postgres"], "node.js": ["nodejs", "node js"],
    "c++": ["cpp"], "c#": ["c sharp", "c-sharp"],
}
SKILL_STOPWORDS = {
    "data", "machine", "learning", "computer", "science", "computer science", "personal", "profile",
    "project", "projects", "analysis", "basic", "undergraduate", "academic", "demo",
    "visualization", "evaluation", "precision", "recall", "structures", "rest",
    "c+", "technology", "technical", "skills", "experience", "knowledge", "management",
    "communication", "team", "work", "working", "ability", "development", "business",
    "services", "solutions", "support", "system", "systems", "process", "planning",
    "reporting", "operations", "engineering", "software", "application", "customer",
    "quality", "training", "research", "tools", "using", "related", "other", "various",
    "new", "full", "time", "role", "roles", "job", "jobs", "company", "responsible",
    "requirements", "required", "preferred", "candidate", "good", "strong"}
MERGE = {"continuous integration": "continuous integration/continuous delivery", "net": ".net",
         "sap s hana": "sap hana", "react": "react.js", "reactjs": "react.js", "react js": "react.js",
         "bi": "business intelligence", "ai": "artificial intelligence"}

def _canonical(s):
    x = re.sub(r"\s+", " ", str(s).lower()).strip()
    for canonical, aliases in SKILL_ALIASES.items():
        if x == canonical or x in aliases:
            return canonical
    return x

def add_skills(df, min_it_listings=10):
    clean = lambda xs: list(dict.fromkeys(s for s in map(_canonical, xs) if s and s not in SKILL_STOPWORDS and len(s) > 1))
    df["skills_list"] = df["skills_list"].map(clean)
    counts = Counter(s for xs in df["skills_list"] for s in set(xs))
    rare = {s for s, c in counts.items() if c < 2}
    df["skills_list"] = df["skills_list"].map(lambda xs: [s for s in xs if s not in rare])
    df["skills_list"] = df["skills_list"].map(lambda xs: list(dict.fromkeys(MERGE.get(s, s) for s in xs)))
    df["skills_text"] = df["skills_list"].map(" ".join)
    it_counts = Counter(s for xs in df.loc[df["family"].ne(OTHER), "skills_list"] for s in set(xs))
    vocab = {s for s, c in it_counts.items() if c >= min_it_listings}
    print("IT skill vocabulary:", len(vocab))
    return df, vocab

# ------------------------------------------------------------------ 4) IT detector (skills only, out-of-fold)
def build_detector(df, n_splits=5):
    X = CountVectorizer(analyzer=lambda x: x, binary=True, min_df=5).fit_transform(df["skills_list"])
    y = df["family"].ne(OTHER).to_numpy().astype(int)
    groups = df["companyName"].fillna("unk")
    p = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits).split(X, y, groups):
        p[va] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return pd.Series(p, index=df.index, name="p_it_skills")

def load_frozen_detector(df, path):
    """Use the SAVED detector scores (the ones the 80 + 30 human labels were stratified on).
    Re-computing them can shift ~1% of tier boundaries, because GroupKFold's fold assignment
    differs between scikit-learn versions; the frozen file keeps the validated estimates valid."""
    saved = pd.read_pickle(path).set_index("jobId")["p_it_skills"]
    p = saved.reindex(df["jobId"].astype(str).values)
    if p.isna().any():
        raise ValueError(f"{int(p.isna().sum())} listings have no frozen detector score")
    return pd.Series(p.values, index=df.index, name="p_it_skills")

# ------------------------------------------------------------------ 5) extended tier
def build_extended(df, p_it):
    df = df.copy(); df["p_it_skills"] = p_it.values
    real = sorted(f for f in df["family"].unique() if f not in (OTHER, AMBIG))
    train = df[df["family"].isin(real)]
    cv = CountVectorizer(analyzer=lambda x: x, binary=True, min_df=3)
    tf = TfidfVectorizer(max_features=20000, min_df=5, sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
    X = hstack([cv.fit_transform(train["skills_list"]), tf.fit_transform(train["description_clean"])]).tocsr()
    model = LogisticRegression(max_iter=1000).fit(X, train["family"])
    prior = train["family"].value_counts(normalize=True).reindex(model.classes_).to_numpy()

    df["family_final"] = df["family"]
    df["family_source"] = np.where(df["family"].isin(real), "rule", np.where(df["family"].eq(AMBIG), "ambiguous", "none"))
    df["pred_conf"] = np.nan
    df["it_tier"] = "non_IT_or_unknown"
    df.loc[df["family"].isin(real), "it_tier"] = "core_rule"
    df.loc[df["family"].eq(AMBIG), "it_tier"] = "core_ambiguous"

    cand = df[df["family"].eq(OTHER) & (df["p_it_skills"] >= IT_CUTOFF)]
    Xc = hstack([cv.transform(cand["skills_list"]), tf.transform(cand["description_clean"])]).tocsr()
    adj = model.predict_proba(Xc) / (prior ** ALPHA)
    adj = adj / adj.sum(axis=1, keepdims=True)
    df.loc[cand.index, "family_final"] = model.classes_[adj.argmax(1)]
    df.loc[cand.index, "family_source"] = "predicted"
    df.loc[cand.index, "pred_conf"] = adj.max(1)
    df.loc[cand.index, "it_tier"] = np.where(cand["p_it_skills"] >= HIGH_CUTOFF, "extended_high", "extended_mid")
    print(df["it_tier"].value_counts().to_string())
    return df
