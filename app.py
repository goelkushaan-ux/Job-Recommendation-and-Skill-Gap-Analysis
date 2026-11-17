"""
IT Career Recommender for India - Streamlit prototype.

Run locally:   pip install -r requirements.txt   then   streamlit run app.py
Needs in the same folder: app.py, recommender.py, career_utils.py, career_pipeline.py, app_data.pkl
"""
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
from recommender import CareerRecommender

st.set_page_config(page_title="IT Career Recommender (India)", page_icon="🧭", layout="wide")

@st.cache_resource(show_spinner="Loading job-market data and fitting the model...")
def load():
    d = pickle.load(open(Path(__file__).with_name("app_data.pkl"), "rb"))
    return CareerRecommender(d["core"], d["market"], d["vocab"]), d["skill_freq"]

rec, skill_options = load()

st.title("🧭 IT Career Recommender for India")
st.caption("Built from ~94,000 Indian job listings (2025). Tell us your skills - we suggest IT career families, "
           "the skills you are missing for each, and an order to learn them in.")

left, right = st.columns([3, 2])
with left:
    text = st.text_area("Paste your resume text or describe what you know", height=170,
                        placeholder="e.g. 2 years of Python, pandas and SQL, some AWS and Docker; built ML models with scikit-learn.")
    picked = st.multiselect("...or pick skills from the list", options=skill_options)
    go = st.button("Recommend careers", type="primary")
with right:
    st.info("**How to read the results**\n\n"
            "- *Match* is the model's probability that jobs with your skills belong to that family.\n"
            "- *Lift* compares it with how common the family is: x10 means it fits you ten times more than a random listing.\n"
            "- *Skills to learn* come from the 200 real listings most similar to your profile.")

if go:
    skills = list(dict.fromkeys(rec.skills_from_text(text) + picked))
    known = rec.known(skills)
    st.subheader("Skills we recognised")
    if not known:
        st.warning("No known skills found. Add a few skills (e.g. python, sql, java) and try again.")
        st.stop()
    st.write(", ".join(f"`{s}`" for s in known))
    if len(known) < 3:
        st.warning("Fewer than 3 skills recognised - the suggestions will be rough. Add more skills for better results.")

    results = rec.recommend(skills, top_k=3, min_prob=0.10)
    if not results:
        st.warning("No family reached 10% match. Try adding more skills.")
        st.stop()

    for n, r in enumerate(results, 1):
        st.divider()
        st.subheader(f"#{n}  {r['family']}")
        a, b = st.columns(2)
        a.metric("Match", f"{r['prob'] * 100:.0f}%")
        b.metric("Lift vs. a random listing", f"x{r['lift']:.1f}")
        st.progress(min(1.0, r["prob"]))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Skills to learn next** (% of similar listings that ask for it)")
            if r["gap"]:
                gap = pd.DataFrame(r["gap"], columns=["skill", "% of similar listings"]).set_index("skill")
                st.bar_chart(gap)
            else:
                st.write("You already cover the common skills of similar listings.")
        with c2:
            if "roadmap" in r:
                st.markdown("**Suggested learning order** (prerequisites first)")
                st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(r["roadmap"], 1)))
        with st.expander(f"Real {r['family']} listings closest to your skills"):
            st.dataframe(rec.similar_jobs(skills, r["family"]), hide_index=True)

with st.expander("How this works and what it cannot do"):
    st.markdown("""
- **Data:** a 2025 snapshot of Indian job listings (about 22,900 labelled by job title, plus about 10,700 more that a
  skills-based detector found). It is a snapshot, not today's market.
- **Model:** a logistic regression on skills. On companies it had never seen it put the right family first for **~80%** of
  listings and in its top three for **~94%** (skills only); the family labels themselves agree with a human reviewer ~85% of the time.
- **Weak spots:** *IT Architect*, *Frontend / Web* and *IT Support* are the least reliable families. Roles like "Application Lead"
  hide their domain in the title, so skills matter more than titles.
- **Not included yet:** your interests and salary. This is a research prototype, not career advice.
""")
