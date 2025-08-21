from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(news_list):
   """
   Analyze sentiment of news headlines.
   Returns a sentiment score and label ( Bullish/Bearish/Neutral).
   """
   if not news_list or "error" in news_list[0]:
       return {"sentiment" : "Unknown", "score": 0.0}
  
   scores =[]
   for news in news_list:
       vs = analyzer.polarity_scores(news["title"])
       scores.append(vs["compound"])


   avg_score = sum(scores)/ len(scores) if scores else 0
   sentiment = "Bullish" if avg_score > 0.05 else "Bearish" if avg_score < 0.05 else "Neutral"


   return{"sentiment": sentiment, "score": round(avg_score, 3)}


