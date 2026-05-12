"""
Integration tests for feature engineering pipeline.

Tests feature building using real data processing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.builder import TimeSeriesBuilder


@pytest.mark.integration
class TestFeatureBuilderIntegration:
    """Integration tests for feature builder with real data."""
    
    def test_build_features_from_fixture_data(self, fixture_price_data, fixture_sentiment_data, tmp_path):
        """Test: Feature building with real fixture data."""
        builder = TimeSeriesBuilder()
        
        # Convert fixture data to DataFrames
        price_df = fixture_price_data.copy()
        sentiment_df = fixture_sentiment_data.copy()
        
        # Ensure date columns are strings
        if 'Date' in price_df.columns:
            price_df['date'] = price_df['Date']
        if not isinstance(price_df['date'].iloc[0], str):
            price_df['date'] = price_df['date'].astype(str)
        
        if not isinstance(sentiment_df['date'].iloc[0], str):
            sentiment_df['date'] = sentiment_df['date'].astype(str)
        
        # Build features
        features_df = builder.build_features(
            price_df=price_df,
            sentiment_df=sentiment_df,
            ticker='AAPL'
        )
        
        # Verify output structure
        assert isinstance(features_df, pd.DataFrame), "Should return DataFrame"
        assert len(features_df) > 0, "Should have feature records"
        
        # Verify required columns
        required_cols = ['date', 'ticker', 'close', 'volume', 'returns']
        for col in required_cols:
            assert col in features_df.columns, f"Should have {col} column"
        
        # Verify data types
        assert pd.api.types.is_numeric_dtype(features_df['close']), \
            "Close should be numeric"
        assert pd.api.types.is_numeric_dtype(features_df['returns']), \
            "Returns should be numeric"
    
    def test_technical_indicators(self, fixture_price_data):
        """Test: Technical indicator calculation."""
        builder = TimeSeriesBuilder()
        
        price_df = fixture_price_data.copy()
        if 'Date' in price_df.columns:
            price_df['date'] = price_df['Date']
        
        # Calculate technical indicators
        features_df = builder.add_technical_indicators(price_df)
        
        # Verify moving averages
        if 'ma_7' in features_df.columns:
            assert features_df['ma_7'].notna().any(), "Should have 7-day MA"
        
        if 'ma_30' in features_df.columns:
            # MA_30 might have NaN for first 29 days
            assert features_df['ma_30'].notna().sum() > 0, "Should have some 30-day MA values"
        
        # Verify volatility
        if 'volatility' in features_df.columns:
            assert (features_df['volatility'] >= 0).all(), \
                "Volatility should be non-negative"
    
    def test_sentiment_aggregation(self, fixture_sentiment_data):
        """Test: Sentiment feature aggregation."""
        builder = TimeSeriesBuilder()
        
        sentiment_df = fixture_sentiment_data.copy()
        
        # Aggregate sentiment by date
        sentiment_agg = builder.aggregate_sentiment(sentiment_df)
        
        # Verify aggregation
        assert isinstance(sentiment_agg, pd.DataFrame), "Should return DataFrame"
        assert 'date' in sentiment_agg.columns, "Should have date column"
        assert 'sentiment_mean' in sentiment_agg.columns or 'vader_compound' in sentiment_agg.columns, \
            "Should have sentiment scores"
        
        # Verify aggregation reduces records
        assert len(sentiment_agg) <= len(sentiment_df), \
            "Aggregation should reduce or maintain record count"
    
    def test_label_generation(self, fixture_price_data):
        """Test: Label generation for supervised learning."""
        builder = TimeSeriesBuilder()
        
        price_df = fixture_price_data.copy()
        if 'Date' in price_df.columns:
            price_df['date'] = price_df['Date']
        if 'Close' in price_df.columns:
            price_df['close'] = price_df['Close']
        
        # Generate labels
        labeled_df = builder.generate_labels(price_df, horizon=1)
        
        # Verify labels
        assert 'label' in labeled_df.columns, "Should have label column"
        assert labeled_df['label'].isin([0, 1]).all(), \
            "Labels should be binary (0 or 1)"
        
        # Verify label distribution (should have both classes ideally)
        unique_labels = labeled_df['label'].unique()
        assert len(unique_labels) > 0, "Should have at least one label class"


@pytest.mark.integration
class TestFeaturePipeline:
    """Integration tests for complete feature engineering pipeline."""
    
    def test_end_to_end_feature_pipeline(self, fixture_price_data, fixture_sentiment_data, tmp_path):
        """Test: Complete feature engineering pipeline."""
        builder = TimeSeriesBuilder()
        
        # Prepare data
        price_df = fixture_price_data.copy()
        sentiment_df = fixture_sentiment_data.copy()
        
        if 'Date' in price_df.columns:
            price_df['date'] = price_df['Date']
        
        # Ensure date columns are strings
        if not isinstance(price_df['date'].iloc[0], str):
            price_df['date'] = price_df['date'].astype(str)
        if not isinstance(sentiment_df['date'].iloc[0], str):
            sentiment_df['date'] = sentiment_df['date'].astype(str)
        
        # 1. Build features
        features_df = builder.build_features(
            price_df=price_df,
            sentiment_df=sentiment_df,
            ticker='AAPL'
        )
        
        # 2. Add technical indicators
        features_df = builder.add_technical_indicators(features_df)
        
        # 3. Generate labels
        features_df = builder.generate_labels(features_df, horizon=1)
        
        # 4. Verify complete feature set
        assert len(features_df) > 0, "Should have feature records"
        assert 'label' in features_df.columns, "Should have labels"
        assert 'returns' in features_df.columns, "Should have returns"
        
        # 5. Save features
        output_file = tmp_path / "features_AAPL.parquet"
        features_df.to_parquet(output_file, index=False)
        
        # 6. Verify output
        assert output_file.exists(), "Output file should be created"
        
        # 7. Load and verify
        loaded_df = pd.read_parquet(output_file)
        assert len(loaded_df) == len(features_df), "Should preserve all records"
        assert 'label' in loaded_df.columns, "Should have labels"
    
    def test_train_test_split(self, fixture_features_data):
        """Test: Chronological train/test split."""
        builder = TimeSeriesBuilder()
        
        features_df = fixture_features_data.copy()
        
        # Ensure date column exists
        if 'date' not in features_df.columns and 'Date' in features_df.columns:
            features_df['date'] = features_df['Date']
        
        # Split data
        train_df, test_df = builder.train_test_split(
            features_df,
            test_size=0.2
        )
        
        # Verify split
        assert len(train_df) > 0, "Should have training data"
        assert len(test_df) > 0, "Should have test data"
        assert len(train_df) + len(test_df) == len(features_df), \
            "Should preserve all records"
        
        # Verify chronological order (train comes before test)
        if 'date' in train_df.columns and 'date' in test_df.columns:
            train_max_date = train_df['date'].max()
            test_min_date = test_df['date'].min()
            assert train_max_date <= test_min_date, \
                "Train data should come before test data"
    
    def test_feature_scaling(self, fixture_features_data):
        """Test: Feature scaling/normalization."""
        builder = TimeSeriesBuilder()
        
        features_df = fixture_features_data.copy()
        
        # Select numeric columns for scaling
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Scale features
            scaled_df, scaler = builder.scale_features(
                features_df,
                columns=numeric_cols[:3]  # Scale first 3 numeric columns
            )
            
            # Verify scaling
            assert isinstance(scaled_df, pd.DataFrame), "Should return DataFrame"
            assert scaler is not None, "Should return scaler object"
            
            # Verify scaled values are in reasonable range
            for col in numeric_cols[:3]:
                if col in scaled_df.columns:
                    # Scaled values should typically be in [-3, 3] range for standard scaling
                    assert scaled_df[col].abs().max() < 10, \
                        f"Scaled {col} should be in reasonable range"
    
    def test_sequence_generation(self, fixture_features_data):
        """Test: Sequence generation for time series models."""
        builder = TimeSeriesBuilder()
        
        features_df = fixture_features_data.copy()
        
        # Ensure we have enough data
        if len(features_df) < 10:
            pytest.skip("Not enough data for sequence generation")
        
        # Generate sequences
        sequences, labels = builder.create_sequences(
            features_df,
            sequence_length=5,
            feature_cols=['close', 'volume']
        )
        
        # Verify sequences
        assert sequences.shape[0] > 0, "Should generate sequences"
        assert sequences.shape[1] == 5, "Sequence length should be 5"
        assert sequences.shape[2] == 2, "Should have 2 features"
        
        # Verify labels
        assert len(labels) == len(sequences), \
            "Should have one label per sequence"
