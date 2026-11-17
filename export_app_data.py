"""
export_app_data.py - makes the small file the app needs (app_data.pkl, a few MB instead of 147 MB).

Run once in Colab after notebook 01:
    %run "/content/drive/MyDrive/Job recommendation nd skill gap analysis/export_app_data.py"
"""
import os, pickle
from collections import Counter
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("CAREER_BASE", "/content/drive/MyDrive/Job recommendation nd skill gap analysis"))
ext = pd.read_pickle(BASE / "processed" / "jobs_final.pkl")
vocab = pickle.load(open(BASE / "processed" / "skill_vocab_final.pkl", "rb"))

core = ext[ext["family_source"].eq("rule")][["skills_list", "family"]].reset_index(drop=True)
market = ext[ext["family_source"].isin(["rule", "predicted"])][
    ["skills_list", "family_final", "title", "companyName", "city", "experience"]].reset_index(drop=True)
freq = Counter(s for sk in market["skills_list"] for s in set(sk) if s in vocab)
skill_freq = [s for s, _ in freq.most_common(500)]

out = BASE / "app_data.pkl"
pickle.dump({"core": core, "market": market, "vocab": vocab, "skill_freq": skill_freq}, open(out, "wb"))
print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB) | core {len(core):,} | market {len(market):,} | skills {len(vocab):,}")
