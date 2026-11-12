"""Text cleaning used inside the TF-IDF vectorizer (importable so the saved model can be loaded)."""

import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "about", "to", "from",
    "in", "on", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "as", "has", "have", "had", "will", "would", "can", "could", "he", "she", "they", "we", "you", "i",
}


def clean_text(text):
    text = str(text)
    exclamations = " exclamationmark" * min(text.count("!"), 3)       # keep a signal of sensational style
    caps = " allcapsword" * min(len(re.findall(r"\b[A-Z]{4,}\b", text)), 3)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)            # links
    text = re.sub(r"<.*?>", " ", text)                                # html tags
    text = re.sub(r"[^a-z\s]", " ", text)                             # numbers / punctuation
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words) + exclamations + caps
