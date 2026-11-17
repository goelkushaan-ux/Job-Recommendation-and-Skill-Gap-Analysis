# 🧭 IT Career Recommender for India

A skills-based career recommender and skill-gap analyser for the Indian IT job market.
You enter your skills (paste resume text or pick from a list) and the app suggests:

1. **Career families** that fit your profile, with a match probability
2. **Missing skills** for each family, based on real listings similar to your profile
3. **A learning order** for those skills, based on how often employers ask for them

Built from about 94,000 Indian job listings (2025 snapshot).

> This is a research prototype, not career advice.

## Features

- Free-text skill extraction from resume text using a skills dictionary
- Multi-select skill picker
- **Match**: the model's probability that jobs with your skills belong to a career family
- **Lift**: how much more likely the family is for you compared to a random listing (x10 means ten times more than random)
- **Skills to learn**: taken from the 200 real listings most similar to your profile

## How it works

| Part | Details |
|---|---|
| Data | 2025 snapshot of Indian job listings: about 22,900 labelled by job title, plus about 10,700 more found by a skills-based IT detector. Extended with listings collected from the Adzuna API |
| Model | Logistic regression on skill features (scikit-learn) |
| Gap recommender | Cosine similarity to find the most similar real listings, then rank the skills you are missing |
| App | Streamlit |

### Results

- On **companies the model had never seen**, the right family was ranked first for about **80%** of listings and in the top three for about **94%** (skills only).
- Family labels agree with a human reviewer about **85%** of the time (102 of 120 checked listings).

### Limitations

- `IT Architect`, `Frontend / Web` and `IT Support` are the least reliable families.
- Roles like "Application Lead" hide their domain in the title, so skills matter more than titles.
- Interests and salary are not included yet.
- The data is a 2025 snapshot, not today's market.

## Project structure

```
app.py                      Streamlit app
recommender.py              CareerRecommender class (model + skill gap + roadmap)
career_utils.py             Skill extractor and helpers
career_pipeline.py          Data cleaning and labelling pipeline
export_app_data.py          Builds the small app_data.pkl used by the app
adzuna_daily_collect.py     Collects new listings from the Adzuna API
app_data.pkl                Small data file the app needs (about 5 MB)
requirements.txt

00_exploration_archive.ipynb   Early exploration
01_data_pipeline.ipynb         Data pipeline
02_models_and_evaluation.ipynb Models and evaluation
processed/results/             Evaluation outputs (CSV / JSON)
```

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/career-recommender.git
cd career-recommender

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Data not included in this repo

The large data files (raw dataset, processed job tables, O*NET files and Adzuna raw downloads) are not stored here because of GitHub's file size limit. The app only needs `app_data.pkl`, which is included.

Data sources:
- Indian Job Market Dataset 2025 (Excel)
- [Adzuna API](https://developer.adzuna.com/) job listings
- [O*NET](https://www.onetcenter.org/database.html) occupation data

To use the Adzuna collector, set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` as environment variables. Never commit API keys.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, Streamlit

## Author

YOUR NAME - [GitHub](https://github.com/YOUR_USERNAME)
