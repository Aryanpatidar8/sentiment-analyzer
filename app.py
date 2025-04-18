from flask import Flask, render_template, request
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

# 🔹 Single Tweet Route
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    score = analyzer.polarity_scores(tweet)['compound']

    if score > 0:
        sentiment = "Positive 😊"
    elif score < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return render_template('index.html', prediction_text=f"Tweet Sentiment: {sentiment}")

# 🔹 Bulk Text Route with Visuals
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['text_block']
    sentences = text.split('.')  # Naive sentence splitting
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    combined_text = ""

    for sentence in sentences:
        if sentence.strip():
            combined_text += " " + sentence
            score = analyzer.polarity_scores(sentence)['compound']
            if score > 0:
                sentiment_counts["Positive"] += 1
            elif score < 0:
                sentiment_counts["Negative"] += 1
            else:
                sentiment_counts["Neutral"] += 1

    # 📊 Sentiment Chart
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=['green', 'red', 'grey'])
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join("static", "sentiment_chart.png"))
    plt.close()

    # ☁️ Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    wordcloud.to_file(os.path.join("static", "wordcloud.png"))

    return render_template("index.html",
                           sentiment_counts=sentiment_counts,
                           show_results=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
