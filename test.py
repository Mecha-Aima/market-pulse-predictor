import requests
api_key = "GQSFFM4IQA4FQIC5"
url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={api_key}"
response = requests.get(url)
print(response.json())  # Should return news data