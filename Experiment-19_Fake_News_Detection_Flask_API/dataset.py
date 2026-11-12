"""
Dataset loader for Fake News Detection.

Priority:
  1. Kaggle "Fake and Real News" dataset   -> data/Fake.csv + data/True.csv
     https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  2. "fake_or_real_news.csv" (columns title, text, label = FAKE/REAL) -> data/fake_or_real_news.csv
  3. WELFake dataset                       -> data/WELFake_Dataset.csv  (label 1 = real, 0 = fake as in the source)
  4. Otherwise a built-in sample dataset is generated (data/news_sample.csv) so the project runs offline.
     It imitates the writing style of real reporting vs. fake/clickbait news. For your final
     submission, download one of the real datasets above for meaningful results.
Returns a DataFrame with columns: text, label  (label: 1 = FAKE, 0 = REAL)
"""

import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

TOPICS = ["the new health policy", "the election results", "the vaccine programme", "the national budget",
          "the climate summit", "the central bank", "the education reform", "the stock market",
          "the space mission", "the trade agreement", "the city metro project", "the cricket board",
          "a new smartphone", "the electricity tariff", "the water supply scheme", "a celebrity wedding"]
OFFICIALS = ["the health minister", "the finance ministry", "a government spokesperson", "the election commission",
             "the World Health Organization", "university researchers", "the police department", "the company CEO",
             "the Reserve Bank", "an independent auditor"]
SOURCES = ["Reuters", "PTI", "Associated Press", "BBC", "The Hindu", "an official press release",
           "a peer-reviewed study", "the annual report"]
NUMBERS = ["2.4 percent", "1.2 million", "340 crore", "15 districts", "7 percent", "12,000 people", "3 years"]

REAL_TEMPLATES = [
    "{official} said on {day} that {topic} will be reviewed next month, according to {source}.",
    "According to {source}, {topic} recorded a change of {number} compared with last year.",
    "{official} confirmed that a committee has been formed to study {topic}. A report is expected by {month}.",
    "Data released by {source} shows {topic} affected about {number}, {official} told reporters on {day}.",
    "In a statement, {official} said the details of {topic} would be published after consultation with experts.",
    "{source} reported that {topic} was discussed in parliament on {day}; opposition members asked for more data.",
    "Officials said {topic} is on schedule. {official} added that funding of {number} has been approved.",
    "A study cited by {source} found mixed results for {topic}, and researchers said more evidence is needed.",
]
FAKE_TEMPLATES = [
    "SHOCKING: {official} secretly admits {topic} is a total scam!!! Share before they delete this!",
    "You won't BELIEVE what they are hiding about {topic}. Doctors HATE this one trick!",
    "BREAKING!!! {topic} is a plot to control your mind, insiders reveal. Mainstream media is silent!",
    "Leaked video PROVES {official} lied about {topic}. Forward to everyone you know!!!",
    "They don't want you to know: {topic} will make you {number} richer overnight. Click now!",
    "Miracle cure discovered in {topic} - big pharma is FURIOUS. 100% guaranteed, no side effects!",
    "URGENT: {topic} cancelled forever, says anonymous source. Government in total panic!!!",
    "EXPOSED: the truth about {topic} that {official} tried to bury. Wake up people!",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["March", "June", "September", "December"]


def _sample_dataset(n_per_class=700, seed=1):
    rng = random.Random(seed)

    def fill(t):
        return t.format(official=rng.choice(OFFICIALS), topic=rng.choice(TOPICS), source=rng.choice(SOURCES),
                        number=rng.choice(NUMBERS), day=rng.choice(DAYS), month=rng.choice(MONTHS))

    rows = []
    for _ in range(n_per_class):
        real = " ".join(fill(rng.choice(REAL_TEMPLATES)) for _ in range(rng.randint(2, 4)))
        fake = " ".join(fill(rng.choice(FAKE_TEMPLATES)) for _ in range(rng.randint(1, 3)))
        # make it harder, like real data:
        # 30% of fake news is written in a calm, "official" tone with one false claim inside
        if rng.random() < 0.30:
            claim = fill(rng.choice(FAKE_TEMPLATES)).lower().replace("!", "")
            fake = fill(rng.choice(REAL_TEMPLATES)).replace("according to", "an unnamed insider claims") + " " + claim
        # 20% of real news has a dramatic headline
        if rng.random() < 0.20:
            real = rng.choice(["BREAKING: ", "Big shock for markets! ", "Explained: the truth about "]) + real
        # 3% annotation errors (real datasets contain mislabelled articles)
        if rng.random() < 0.03:
            real, fake = fake, real
        rows += [{"text": real, "label": 0}, {"text": fake, "label": 1}]
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_news():
    if (DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists():
        fake = pd.read_csv(DATA_DIR / "Fake.csv").assign(label=1)
        real = pd.read_csv(DATA_DIR / "True.csv").assign(label=0)
        # True.csv articles start with "CITY (Reuters) -" which would leak the label - remove it
        real["text"] = real["text"].str.replace(r"^.*?\(Reuters\)\s*-\s*", "", regex=True)
        df = pd.concat([fake, real])
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        print("Using Kaggle Fake/True news dataset")
    elif (DATA_DIR / "fake_or_real_news.csv").exists():
        df = pd.read_csv(DATA_DIR / "fake_or_real_news.csv")
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        df["label"] = (df["label"].str.upper() == "FAKE").astype(int)
        print("Using fake_or_real_news dataset")
    elif (DATA_DIR / "WELFake_Dataset.csv").exists():
        df = pd.read_csv(DATA_DIR / "WELFake_Dataset.csv")
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        df["label"] = 1 - df["label"]
        print("Using WELFake dataset")
    else:
        sample = DATA_DIR / "news_sample.csv"
        if not sample.exists():
            DATA_DIR.mkdir(exist_ok=True)
            _sample_dataset().to_csv(sample, index=False)
        df = pd.read_csv(sample)
        print("Using built-in sample dataset (download the Kaggle dataset for real results)")
    return df[["text", "label"]].dropna().drop_duplicates(subset="text").reset_index(drop=True)
