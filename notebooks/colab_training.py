# %% [markdown]
# # Market Pulse Predictor - Colab Training
# 
# This notebook trains **9 models** (3 architectures × 3 tasks) on Colab GPU with MLflow tracking.
# 
# ## What This Notebook Does:
# - ✅ Trains RNN, LSTM, and GRU models
# - ✅ For 3 tasks: Direction, Return, Volatility
# - ✅ Implements early stopping (patience=7)
# - ✅ Logs all experiments to Railway MLflow
# - ✅ Registers best models in Model Registry
# - ✅ Exports all trained models as .pt files
# - ✅ Creates downloadable artifacts package
# 
# ## Setup Instructions:
# 1. Upload `colab_upload/features.tar.gz` to Google Drive
# 2. Set your Railway MLflow URL below
# 3. Run all cells (~1-2 hours on T4 GPU)
# 4. Download `trained_models.zip` at the end
# 
# ## Expected Training Time:
# - Per model: 5-15 minutes
# - Total (9 models): 1-2 hours
# - With early stopping: May finish faster

# %%
# Install dependencies
!pip install -q torch mlflow pyyaml scikit-learn numpy pandas boto3 psycopg2-binary

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/market-pulse-predictor.git
%cd market-pulse-predictor

# %%
# Set environment variables for Railway MLflow
import os

# ⚠️ REPLACE WITH YOUR RAILWAY MLFLOW URL
# Get this from: railway domain (or Railway dashboard)
os.environ['MLFLOW_TRACKING_URI'] = 'https://your-app.up.railway.app'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'market_pulse'

# AWS credentials for S3 artifact storage
# These should match your Railway environment variables
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

print(f"✅ MLflow Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
print(f"✅ Experiment: {os.environ['MLFLOW_EXPERIMENT_NAME']}")

# %%
# Copy feature data from Google Drive
!mkdir -p data/features

# Option 1: If you uploaded the tar.gz
!tar -xzf /content/drive/MyDrive/market-pulse-data/features.tar.gz -C .

# Option 2: If you uploaded the folder directly
# !cp -r /content/drive/MyDrive/market-pulse-data/features/* data/features/

# Verify files exist
!ls -lh data/features/
print("\n✅ Feature data loaded")

# %%
# Verify GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# %%
# Update params.yaml for GPU training with early stopping
import yaml

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configure for production training
params['training']['device'] = 'cuda'
params['training']['epochs'] = 50
params['training']['batch_size'] = 64
params['training']['learning_rate'] = 0.001
params['training']['hidden_size'] = 128
params['training']['num_layers'] = 2
params['training']['dropout'] = 0.2
params['training']['early_stopping_patience'] = 7  # Stop if no improvement for 7 epochs

with open('params.yaml', 'w') as f:
    yaml.dump(params, f)

print("✅ Training configuration:")
print(f"   Device: {params['training']['device']}")
print(f"   Max Epochs: {params['training']['epochs']}")
print(f"   Batch Size: {params['training']['batch_size']}")
print(f"   Early Stopping Patience: {params['training']['early_stopping_patience']}")
print(f"   Hidden Size: {params['training']['hidden_size']}")
print(f"   Num Layers: {params['training']['num_layers']}")

# %% [markdown]
# ## Test MLflow Connection
# 
# Verify we can connect to Railway MLflow before training

# %%
import mlflow
from mlflow.tracking import MlflowClient

# Test connection
try:
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    client = MlflowClient()
    experiments = client.search_experiments()
    print(f"✅ Connected to MLflow!")
    print(f"   Found {len(experiments)} experiments")
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.set_experiment(os.environ['MLFLOW_EXPERIMENT_NAME'])
        print(f"✅ Using experiment: {experiment.name}")
    except Exception as e:
        print(f"⚠️  Could not create experiment: {e}")
        print("   Will use default experiment")
except Exception as e:
    print(f"❌ MLflow connection failed: {e}")
    print("   Check your MLFLOW_TRACKING_URI")
    raise

# %% [markdown]
# ## Train All Models
# 
# This will train 9 models total:
# - 3 architectures: RNN, LSTM, GRU
# - 3 tasks: Direction, Return, Volatility
# 
# Each model includes:
# - ✅ Early stopping (patience=7)
# - ✅ Best model checkpoint saving
# - ✅ MLflow experiment logging
# - ✅ Model registry registration
# - ✅ Metrics tracking (accuracy, F1, RMSE)

# %%
# Train all combinations
from src.training.run_training import run_training_pipeline
import yaml
from datetime import datetime
import time

models = ['rnn', 'lstm', 'gru']
tasks = ['direction', 'return', 'volatility']

results = {}
training_start = time.time()

print("="*80)
print("STARTING TRAINING PIPELINE")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Models: {len(models) * len(tasks)}")
print(f"Models: {', '.join(models)}")
print(f"Tasks: {', '.join(tasks)}")
print("="*80)
print()

for i, model in enumerate(models, 1):
    for j, task in enumerate(tasks, 1):
        model_num = (i-1) * len(tasks) + j
        print(f"\n{'='*80}")
        print(f"MODEL {model_num}/9: {model.upper()} - {task.upper()}")
        print(f"{'='*80}")
        
        model_start = time.time()
        
        # Update params
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        params['training']['model'] = model
        params['training']['task'] = task
        with open('params.yaml', 'w') as f:
            yaml.dump(params, f)
        
        try:
            # Train
            result = run_training_pipeline()
            results[f"{model}_{task}"] = result
            
            model_time = time.time() - model_start
            
            print(f"\n✅ COMPLETED {model}_{task} in {model_time/60:.1f} minutes")
            print(f"\nMetrics:")
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            
            # Show training history summary
            history = result['history']
            if 'train_loss' in history:
                epochs_trained = len(history['train_loss'])
                final_train_loss = history['train_loss'][-1]
                final_val_loss = history['val_loss'][-1]
                print(f"\nTraining Summary:")
                print(f"  Epochs: {epochs_trained}/{params['training']['epochs']}")
                print(f"  Final Train Loss: {final_train_loss:.4f}")
                print(f"  Final Val Loss: {final_val_loss:.4f}")
                if epochs_trained < params['training']['epochs']:
                    print(f"  ⚡ Early stopping triggered!")
            
        except Exception as e:
            print(f"\n❌ FAILED {model}_{task}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results[f"{model}_{task}"] = {"error": str(e)}
        
        # Progress update
        elapsed = time.time() - training_start
        avg_time = elapsed / model_num
        remaining = avg_time * (9 - model_num)
        print(f"\n📊 Progress: {model_num}/9 models ({model_num/9*100:.0f}%)")
        print(f"⏱️  Elapsed: {elapsed/60:.1f} min | Estimated remaining: {remaining/60:.1f} min")

total_time = time.time() - training_start
print(f"\n{'='*80}")
print("TRAINING PIPELINE COMPLETE")
print(f"{'='*80}")
print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Successful: {sum(1 for r in results.values() if 'error' not in r)}/9")
print(f"Failed: {sum(1 for r in results.values() if 'error' in r)}/9")

# %%
# Display comprehensive summary of all results
import pandas as pd
import numpy as np

summary = []
for key, result in results.items():
    if 'error' in result:
        continue
    
    model, task = key.split('_', 1)
    metrics = result['metrics']
    history = result['history']
    
    row = {
        'Model': model.upper(),
        'Task': task.capitalize(),
        'Epochs': len(history.get('train_loss', [])),
        'Final Train Loss': history['train_loss'][-1] if 'train_loss' in history else None,
        'Final Val Loss': history['val_loss'][-1] if 'val_loss' in history else None,
    }
    
    # Add task-specific metrics
    if 'accuracy' in metrics:
        row['Accuracy'] = metrics['accuracy']
    if 'f1_score' in metrics:
        row['F1 Score'] = metrics['f1_score']
    if 'rmse' in metrics:
        row['RMSE'] = metrics['rmse']
    
    summary.append(row)

df = pd.DataFrame(summary)

print("\n" + "="*100)
print("TRAINING SUMMARY - ALL MODELS")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# Find best models per task
print("\n🏆 BEST MODELS PER TASK:")
print("="*100)

for task in ['Direction', 'Return', 'Volatility']:
    task_df = df[df['Task'] == task]
    if len(task_df) == 0:
        continue
    
    if task == 'Return':
        # Lower RMSE is better
        best = task_df.loc[task_df['RMSE'].idxmin()]
        print(f"\n{task}: {best['Model']} (RMSE: {best['RMSE']:.4f})")
    else:
        # Higher accuracy is better
        best = task_df.loc[task_df['Accuracy'].idxmax()]
        print(f"\n{task}: {best['Model']} (Accuracy: {best['Accuracy']:.4f}, F1: {best['F1 Score']:.4f})")

print("\n" + "="*100)

# Save summary to CSV
df.to_csv('training_summary.csv', index=False)
print("\n✅ Summary saved to training_summary.csv")

# %% [markdown]
# ## Verify MLflow Model Registry
# 
# Check that all models were registered correctly

# %%
# Check registered models
from mlflow.tracking import MlflowClient

client = MlflowClient()

print("\n" + "="*80)
print("MLFLOW MODEL REGISTRY")
print("="*80)

for task in ['direction', 'return', 'volatility']:
    model_name = f"MarketPulsePredictor-{task}"
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        print(f"\n📦 {model_name}:")
        print(f"   Total versions: {len(versions)}")
        
        # Find production version
        prod_versions = [v for v in versions if v.current_stage == 'Production']
        if prod_versions:
            prod = prod_versions[0]
            print(f"   ✅ Production: Version {prod.version}")
            print(f"      Run ID: {prod.run_id}")
        else:
            print(f"   ⚠️  No production version set")
    except Exception as e:
        print(f"\n❌ {model_name}: {e}")

print("\n" + "="*80)

# %% [markdown]
# ## Package and Download All Artifacts
# 
# Create a comprehensive package with:
# - All trained model .pt files
# - Training summary CSV
# - Model metadata
# - Feature scaler (if exists)

# %% [markdown]
# ## Next Steps
# 
# 1. Download `trained_models.zip`
# 2. Extract to your local `models/` directory
# 3. All experiments are logged to your remote MLflow server
# 4. Models are registered in MLflow Model Registry
# 5. Your API will automatically load the Production models


