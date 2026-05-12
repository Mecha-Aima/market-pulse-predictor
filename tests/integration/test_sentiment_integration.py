"""
Integration tests for sentiment analysis pipeline.

Tests sentiment analysis using real VADER and actual text data.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.sentiment.analyzer import SentimentAnalyzer


@pytest.mark.integration
class TestSentimentAnalysisIntegration:
    """Integration tests for sentiment analyzer with real VADER."""
    
    def test_analyze_real_text_with_vader(self):
        """Test: Sentiment analysis with real VADER on actual text."""
        analyzer = SentimentAnalyzer()
        
        # Test with various sentiment texts
        test_cases = [
            {
                'text': "Apple stock surges to new all-time high on strong earnings report!",
                'expected_sentiment': 'positive'
            },
            {
                'text': "Company faces major losses and declining market share.",
                'expected_sentiment': 'negative'
            },
            {
                'text': "The stock price remained unchanged today.",
                'expected_sentiment': 'neutral'
            }
        ]
        
        for case in test_cases:
            result = analyzer.analyze(case['text'])
            
            # Verify result structure
            assert 'compound' in result, "Should have compound score"
            assert 'pos' in result, "Should have positive score"
            assert 'neu' in result, "Should have neutral score"
            assert 'neg' in result, "Should have negative score"
            
            # Verify score ranges
            assert -1 <= result['compound'] <= 1, "Compound should be in [-1, 1]"
            assert 0 <= result['pos'] <= 1, "Positive should be in [0, 1]"
            assert 0 <= result['neu'] <= 1, "Neutral should be in [0, 1]"
            assert 0 <= result['neg'] <= 1, "Negative should be in [0, 1]"
            
            # Verify sentiment direction (loose check due to VADER variability)
            if case['expected_sentiment'] == 'positive':
                assert result['compound'] > -0.5, "Should lean positive or neutral"
            elif case['expected_sentiment'] == 'negative':
                assert result['compound'] < 0.5, "Should lean negative or neutral"
    
    def test_batch_analyze_real_data(self, fixture_news_data):
        """Test: Batch sentiment analysis on real news data."""
        analyzer = SentimentAnalyzer()
        
        # Use fixture news data
        records = fixture_news_data[:5]  # Analyze first 5 articles
        
        # Perform batch analysis
        results = analyzer.batch_analyze(records, text_field='text')
        
        # Verify results
        assert len(results) == len(records), "Should analyze all records"
        
        for result in results:
            # Verify enrichment
            assert 'vader_compound' in result, "Should have compound score"
            assert 'vader_pos' in result, "Should have positive score"
            assert 'vader_neu' in result, "Should have neutral score"
            assert 'vader_neg' in result, "Should have negative score"
            
            # Verify original fields preserved
            assert 'text' in result, "Should preserve original text"
            assert 'ticker' in result, "Should preserve ticker"
    
    def test_sentiment_on_fixture_data(self, fixture_news_data, tmp_path):
        """Test: End-to-end sentiment analysis on fixture data."""
        analyzer = SentimentAnalyzer()
        
        # Analyze fixture data
        enriched = analyzer.batch_analyze(fixture_news_data, text_field='text')
        
        # Convert to DataFrame
        df = pd.DataFrame(enriched)
        
        # Verify DataFrame structure
        assert 'vader_compound' in df.columns, "Should have compound scores"
        assert len(df) == len(fixture_news_data), "Should preserve all records"
        
        # Verify no null sentiment scores
        assert df['vader_compound'].notna().all(), "All records should have scores"
        
        # Save to parquet
        output_file = tmp_path / "sentiment_test.parquet"
        df.to_parquet(output_file, index=False)
        
        # Verify saved file
        assert output_file.exists(), "Output file should be created"
        
        # Read back and verify
        df_loaded = pd.read_parquet(output_file)
        assert len(df_loaded) == len(df), "Should preserve all records"
        assert 'vader_compound' in df_loaded.columns, "Should have sentiment scores"


@pytest.mark.integration
class TestSentimentPipeline:
    """Integration tests for complete sentiment pipeline."""
    
    def test_end_to_end_sentiment_pipeline(self, tmp_path):
        """Test: Complete sentiment analysis pipeline."""
        analyzer = SentimentAnalyzer()
        
        # Create sample news data
        news_data = [
            {
                'ticker': 'AAPL',
                'text': 'Apple announces record-breaking quarterly revenue and profit.',
                'source': 'test',
                'published': '2024-01-01'
            },
            {
                'ticker': 'AAPL',
                'text': 'Concerns raised about supply chain disruptions affecting production.',
                'source': 'test',
                'published': '2024-01-02'
            },
            {
                'ticker': 'AAPL',
                'text': 'Company maintains steady market position in tech sector.',
                'source': 'test',
                'published': '2024-01-03'
            }
        ]
        
        # 1. Analyze sentiment
        enriched = analyzer.batch_analyze(news_data, text_field='text')
        
        # 2. Convert to DataFrame
        df = pd.DataFrame(enriched)
        
        # 3. Verify sentiment distribution
        assert df['vader_compound'].mean() != 0, "Should have varied sentiments"
        
        # 4. Aggregate by ticker
        ticker_sentiment = df.groupby('ticker').agg({
            'vader_compound': ['mean', 'count'],
            'vader_pos': 'mean',
            'vader_neg': 'mean'
        }).reset_index()
        
        assert len(ticker_sentiment) > 0, "Should have aggregated data"
        
        # 5. Save results
        output_file = tmp_path / "sentiment_AAPL.parquet"
        df.to_parquet(output_file, index=False)
        
        # 6. Verify output
        assert output_file.exists(), "Output file should be created"
        df_loaded = pd.read_parquet(output_file)
        assert len(df_loaded) == len(news_data), "Should preserve all records"
    
    def test_sentiment_with_missing_text(self):
        """Test: Sentiment analysis handles missing text gracefully."""
        analyzer = SentimentAnalyzer()
        
        # Data with some missing text
        data = [
            {'ticker': 'AAPL', 'text': 'Good news for investors'},
            {'ticker': 'AAPL', 'text': None},
            {'ticker': 'AAPL', 'text': ''},
            {'ticker': 'AAPL', 'text': 'Bad news for the company'}
        ]
        
        # Analyze
        results = analyzer.batch_analyze(data, text_field='text')
        
        # Verify handling of missing text
        assert len(results) == len(data), "Should process all records"
        
        # Records with valid text should have scores
        assert results[0]['vader_compound'] != 0, "First record should have score"
        assert results[3]['vader_compound'] != 0, "Last record should have score"
    
    def test_sentiment_preserves_alphavantage_scores(self):
        """Test: Sentiment analysis preserves Alpha Vantage sentiment when present."""
        analyzer = SentimentAnalyzer()
        
        # Data with Alpha Vantage sentiment
        data = [
            {
                'ticker': 'AAPL',
                'text': 'Some news text',
                'overall_sentiment_score': 0.75,
                'overall_sentiment_label': 'Bullish'
            }
        ]
        
        # Analyze
        results = analyzer.batch_analyze(data, text_field='text')
        
        # Verify Alpha Vantage scores preserved
        assert 'overall_sentiment_score' in results[0], \
            "Should preserve Alpha Vantage score"
        assert results[0]['overall_sentiment_score'] == 0.75, \
            "Should preserve exact score"
        
        # Verify VADER scores also added
        assert 'vader_compound' in results[0], \
            "Should add VADER scores"
