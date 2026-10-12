import pandas as pd
import re
import html
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, render_template_string

# 1. Load Data
df = pd.read_csv("merged_cleaned.csv", encoding='utf-8-sig')

# 2. UPDATED: Emoji-Aware Cleaning
def clean_text_preserving_emojis(text):
    text = html.unescape(str(text))
    text = text.lower()
    
    # Remove URLs, Mentions, Hashtags
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    
    # CRITICAL FIX: Instead of removing everything except a-z, 
    # we only remove "weird junk" and keep letters + Emojis.
    # This regex protects the Unicode ranges for emojis.
    text = re.sub(r'[^\w\s\u2600-\u27BF\U0001f300-\U0001f64f\U0001f680-\U0001f6ff]', ' ', text)
    
    return re.sub(r'\s+', ' ', text).strip()

# 3. UPDATED: Tokenization with Slang & Emoji Translation
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def process_features(text):
    # Convert Emoji to text first! 😍 -> "heart_eyes"
    # This gives the Naive Bayes model actual words to learn from.
    text_with_emojis_as_words = emoji.demojize(text, delimiters=(" ", " "))
    
    tokens = word_tokenize(text_with_emojis_as_words)
    
    # Remove underscores from emoji names (e.g., "heart_eyes" -> "heart eyes")
    clean_tokens = [t.replace("_", " ") for t in tokens if t not in stop_words]
    
    return " ".join(clean_tokens)

# Apply processing
df['cleaned'] = df['text'].apply(clean_text_preserving_emojis)
df['processed_text'] = df['cleaned'].apply(process_features)

# 4. Balanced Training (3-Way)
df_pos = df[df['label'] == 'positive']
df_neg = df[df['label'] == 'negative']
df_neu = df[df['label'] == 'neutral']

min_size = min(len(df_pos), len(df_neg), len(df_neu))

df_balanced = pd.concat([
    resample(df_pos, replace=False, n_samples=min_size, random_state=42),
    resample(df_neg, replace=False, n_samples=min_size, random_state=42),
    resample(df_neu, replace=False, n_samples=min_size, random_state=42)
]).sample(frac=1, random_state=42)

# 5. Vectorize & Train
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_balanced['processed_text'])
y = df_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Prediction Logic
def predict_sentiment(new_text):
    cleaned = clean_text_preserving_emojis(new_text)
    processed = process_features(cleaned)
    vector = tfidf.transform([processed])
    return model.predict(vector)[0]

# 7. Modern UI (Bootstrap 5)
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sentiment Analysis AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; font-family: 'Inter', sans-serif; display: flex; align-items: center; }
        .card { border: none; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .btn-primary { background: #764ba2; border: none; padding: 12px; border-radius: 10px; font-weight: 600; }
        .btn-primary:hover { background: #5a3782; }
        .result-box { margin-top: 25px; padding: 20px; border-radius: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
        .sentiment-positive { background-color: #d1e7dd; color: #0f5132; border: 2px solid #0f5132; }
        .sentiment-negative { background-color: #f8d7da; color: #842029; border: 2px solid #842029; }
        .sentiment-neutral { background-color: #e2e3e5; color: #41464b; border: 2px solid #41464b; }
        textarea { border-radius: 10px !important; }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card p-5">
                    <h2 class="text-center mb-4" style="color: #4a4a4a;">Sentiment AI</h2>
                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label text-muted">What's on your mind?</label>
                            <textarea name="post_text" class="form-control" rows="4" placeholder="Type or paste text with emojis here...">{{ text }}</textarea>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Analyze Now</button>
                    </form>

                    {% if sentiment %}
                    <div class="result-box text-center sentiment-{{ sentiment }}">
                        Result: {{ sentiment }}
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    text = ""
    if request.method == "POST":
        text = request.form["post_text"]
        sentiment = predict_sentiment(text)
    return render_template_string(HTML_TEMPLATE, sentiment=sentiment, text=text)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)