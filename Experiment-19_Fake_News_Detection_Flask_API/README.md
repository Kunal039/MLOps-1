# Experiment 19 – Fake News Detection using ML and Flask API

**Sample project · CO3/CO4**

## Aim
To classify news text as FAKE or REAL with NLP and machine learning, and to serve the model through a Flask REST API and web page.

## Data (`dataset.py`)
Real datasets are used automatically if you place them in `data/`:
- Kaggle **Fake.csv + True.csv**: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- `fake_or_real_news.csv`, or `WELFake_Dataset.csv`

If none are present, a built-in sample dataset (1,400 articles) is generated so the project runs offline. It imitates real reporting vs clickbait/fake style, including hard cases and 3% label noise. **Use a real dataset for the final submission.**

## Method
- `text_utils.clean_text`: lowercase, remove URLs/HTML/punctuation/stopwords, and keep style signals (`!`, ALL-CAPS words).
- **TF-IDF** on word 1–2 grams, with sublinear term frequency.
- Classifiers compared: Multinomial Naive Bayes, Logistic Regression, Passive-Aggressive, Linear SVM.
- The script prints the most informative words (transparency) and misclassified examples (error analysis).

## API
| Method | Path | Body |
|---|---|---|
| GET/POST | `/` | Web page |
| POST | `/api/predict` | `{"text": "..."}` → `{"label": "FAKE", "fake_probability": 0.97, "confidence": 0.97}` |
| POST | `/api/batch` | `{"articles": ["...", "..."]}` |
| GET | `/api/health` | Model info |

## Run
```bash
python train.py
python -m pytest test_api.py -v
python app.py
```
Then open http://127.0.0.1:5000.

## Result (sample data)
F1 ≈ 0.98 for all models. Logistic Regression is selected because it gives probabilities. Words such as *said, according, report, data* indicate REAL; words such as *exclamationmark, wake up, truth, tried to bury* indicate FAKE. 6/6 API tests pass.
