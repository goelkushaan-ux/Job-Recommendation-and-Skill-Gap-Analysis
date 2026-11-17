"""
recommender.py - the career recommender (used by notebook 02 and by the Streamlit app).

    rec = CareerRecommender(core_df, market_df, vocab)
    rec.recommend(skills)             -> top families with probability, lift, personalised skill gap, roadmap
    rec.skills_from_text("free text") -> skills found by the dictionary extractor

core_df    : rule-labelled IT listings (clean labels)  -> trains the skills-only family model
market_df  : core + predicted-family listings          -> neighbours for the gap and demand for the roadmap
             (sensitivity check: adding the predicted listings improved the gap recommender)
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from career_utils import SkillExtractor
from career_pipeline import _canonical

GENERIC_SKILLS = {"artificial intelligence", "data science", "algorithms", "data", "consulting",
    "project management", "troubleshooting", "automation", "architecture", "software development",
    "application development", "development methodologies", "technical leadership",
    "design principles", "application design", "web services", "debugging", "analytical"}
GENERIC_WORDS = {"development", "developer", "programming", "engineering", "framework", "skills", "application", "software"}
_tok = lambda s: set(re.findall(r"[a-z0-9#+.]+", s.lower()))

PREREQUISITES = {
    "python": [], "sql": [], "statistics": [], "linear algebra": ["statistics"], "numpy": ["python"],
    "pandas": ["python", "numpy"], "scikit-learn": ["python", "pandas", "statistics"],
    "machine learning": ["python", "statistics", "scikit-learn"],
    "deep learning": ["python", "linear algebra", "machine learning"],
    "pytorch": ["python", "deep learning"], "tensorflow": ["python", "deep learning"],
    "natural language processing": ["python", "machine learning"],
    "generative ai": ["python", "machine learning", "natural language processing"],
    "large language models": ["python", "natural language processing", "deep learning"],
    "power bi": ["sql", "microsoft excel"], "tableau": ["sql"], "git": [], "linux": [],
    "docker": ["linux"], "kubernetes": ["docker", "linux"], "aws": ["linux"], "amazon web services": ["linux"],
    "azure": ["linux"], "google cloud platform": ["linux"], "spark": ["python", "sql"],
    "apache spark": ["python", "sql"], "airflow": ["python", "sql"], "data engineering": ["python", "sql"],
    "mongodb": [], "postgresql": ["sql"], "javascript": [], "html": [], "css": ["html"],
    "react": ["javascript", "html", "css"], "react.js": ["javascript", "html", "css"],
    "node.js": ["javascript"], "java": [], "c++": [], "networking": [], "cybersecurity": ["networking", "linux"],
}

class CareerRecommender:
    def __init__(self, core_df, market_df, vocab, family_col="family", market_family_col="family_final"):
        self.vocab = set(vocab)
        self.extractor = SkillExtractor(self.vocab)
        self.cv = CountVectorizer(analyzer=lambda x: x, binary=True, min_df=3)
        X = self.cv.fit_transform(core_df["skills_list"])
        self.model = LogisticRegression(max_iter=1000).fit(X, core_df[family_col])
        self.prior = core_df[family_col].value_counts(normalize=True)
        self.market = {}
        for fam, sub in market_df.groupby(market_family_col):
            self.market[fam] = (sub, self.cv.transform(sub["skills_list"]))
        self.demand = {fam: Counter(s for sk in sub["skills_list"] for s in set(sk))
                       for fam, (sub, _) in self.market.items()}

    # ---- input
    def skills_from_text(self, text):
        return [_canonical(s) for s in self.extractor.extract(text)]

    def known(self, skills):
        return [s for s in dict.fromkeys(skills) if s in self.cv.vocabulary_]

    # ---- step 1: which families fit these skills?
    def predict_families(self, skills):
        p = self.model.predict_proba(self.cv.transform([sorted(self.known(skills))]))[0]
        out = pd.DataFrame({"family": self.model.classes_, "prob": p})
        out["market_share"] = out["family"].map(self.prior)
        out["lift"] = out["prob"] / out["market_share"]
        return out.sort_values("prob", ascending=False).reset_index(drop=True)

    # ---- step 2: personalised gap = skills of the most similar listings that the person lacks
    def personalized_gap(self, skills, family, k=200, gap_n=8):
        sub, X = self.market[family]
        rs = set(skills); rs_tok = [_tok(s) for s in rs]
        sims = cosine_similarity(self.cv.transform([sorted(self.known(skills))]), X)[0]
        nn = sub.iloc[np.argsort(-sims)[:k]]
        c = Counter(s for sk in nn["skills_list"] for s in set(sk))

        def redundant(s):
            tg = _tok(s); core = tg - GENERIC_WORDS
            return any(tg <= t or (core and core <= t) for t in rs_tok)
        rows = [(s, round(100 * v / len(nn), 1)) for s, v in c.items()
                if s not in rs and s not in GENERIC_SKILLS and s in self.cv.vocabulary_ and len(s) >= 3 and not redundant(s)]
        rows.sort(key=lambda r: -r[1])
        rows = rows[: gap_n * 2]
        toks = [(_tok(s), s) for s, _ in rows]                 # drop a skill that is a strict part of another listed skill
        keep = {s for t, s in toks if not any(t < t2 for t2, s2 in toks if s2 != s)}
        return pd.DataFrame([r for r in rows if r[0] in keep][:gap_n], columns=["skill", "pct_of_similar_listings"])

    # ---- step 3: learning order (prerequisites first, then market demand)
    def roadmap(self, skills, missing, family, max_items=12):
        have = set(skills)
        expanded, changed = {_canonical(s) for s in missing}, True
        while changed:
            changed = False
            for s in list(expanded):
                for pre in PREREQUISITES.get(s, []):
                    pre = _canonical(pre)
                    if pre not in have and pre not in expanded:
                        expanded.add(pre); changed = True
        demand = self.demand.get(family, Counter())
        ordered, remaining = [], set(expanded)
        while remaining:
            ready = [s for s in remaining if all(_canonical(p) not in remaining for p in PREREQUISITES.get(s, []))] or list(remaining)
            ready.sort(key=lambda s: (demand.get(s, 0), s), reverse=True)
            for s in ready:
                ordered.append(s); remaining.discard(s)
        return ordered[:max_items]

    # ---- real listings that look most like the person's skills (for the app)
    def similar_jobs(self, skills, family, n=5, cols=("title", "companyName", "city", "experience")):
        sub, X = self.market[family]
        sims = cosine_similarity(self.cv.transform([sorted(self.known(skills))]), X)[0]
        top = sub.iloc[np.argsort(-sims)[:n]]
        return top[[c for c in cols if c in top.columns]].reset_index(drop=True)

    # ---- everything together
    def recommend(self, skills, top_k=3, min_prob=0.10, gap_n=8):
        fam = self.predict_families(skills)
        result = []
        for i, row in fam[fam["prob"] >= min_prob].head(top_k).iterrows():
            if row["family"] not in self.market:
                continue
            gap = self.personalized_gap(skills, row["family"], gap_n=gap_n)
            item = {"family": row["family"], "prob": float(row["prob"]), "lift": float(row["lift"]),
                    "gap": list(zip(gap["skill"], gap["pct_of_similar_listings"]))}
            if not result and len(gap):
                item["roadmap"] = self.roadmap(skills, gap["skill"].tolist(), row["family"])
            result.append(item)
        return result

    def print_report(self, skills, **kw):
        print(f"Skills recognised: {len(self.known(skills))} of {len(set(skills))}")
        for n, r in enumerate(self.recommend(skills, **kw), 1):
            print(f"\n#{n} {r['family']} | prob {r['prob']:.2f} | lift x{r['lift']:.1f}")
            print("   learn next:", ", ".join(f"{s} ({v}%)" for s, v in r["gap"]))
            if "roadmap" in r:
                print("   roadmap:", " -> ".join(r["roadmap"]))
