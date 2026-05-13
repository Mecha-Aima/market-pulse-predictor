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
# - ✅ Registers best models in Model Registry (alias-based, MLflow 2.x compatible)
# - ✅ Exports all trained models as .pt files
# - ✅ Packages scaler.pkl + feature_columns.json into downloadable artifact
#
# ## Setup Instructions:
# 1. Upload `features.tar.gz` to Google Drive under `market-pulse-data/`
# 2. Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to Colab Secrets (🔑 sidebar)
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
!git clone https://github.com/Mecha-Aima/market-pulse-predictor.git
%cd market-pulse-predictor

# %%
# Set environment variables for Railway MLflow
import os
from google.colab import userdata

# ── MLflow (Railway) ────────────────────────────────────────────────────────
os.environ['MLFLOW_TRACKING_URI'] = 'https://market-pulse-predictor-production.up.railway.app'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'market_pulse'

# ── AWS credentials (S3 artifact storage) ───────────────────────────────────
# ⚠️  DO NOT paste credentials here — use Colab Secrets instead:
#   1. Click the 🔑 key icon in the left sidebar (or Secrets in Runtime menu)
#   2. Add: AWS_ACCESS_KEY_ID   → your key ID
#   3. Add: AWS_SECRET_ACCESS_KEY → your secret key
#   4. Toggle "Notebook access" ON for each secret
os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
# FIX: Do NOT set MLFLOW_S3_ENDPOINT_URL to empty string — boto3 treats '' as a
# literal endpoint and fails. Only set it when using a non-AWS S3-compatible store.
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = ''  ← removed

print(f"✅ MLflow Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
print(f"✅ Experiment: {os.environ['MLFLOW_EXPERIMENT_NAME']}")
print(f"✅ AWS Key ID (first 8 chars): {os.environ['AWS_ACCESS_KEY_ID'][:8]}...")

# %%
# Configure DVC remote credentials
# DVC needs AWS creds exported for dvc push/pull and for resolve_data_version().
# The repo .dvc/config already points to s3://market-pulse-dvc/dvc-store.
!pip install -q dvc dvc-s3

# Write DVC credentials into the local config so `dvc data status` works in Colab
!dvc remote modify --local s3remote access_key_id $AWS_ACCESS_KEY_ID
!dvc remote modify --local s3remote secret_access_key $AWS_SECRET_ACCESS_KEY

# Verify DVC can see the remote (non-fatal if data wasn't pushed)
!dvc remote list
print("✅ DVC remote configured")

# %%
# Copy feature data from Google Drive
!mkdir -p data/features

# Option 1: If you uploaded the tar.gz (recommended — smaller upload)
!tar -xzf /content/drive/MyDrive/market-pulse-data/features.tar.gz -C .

# Option 2: If you uploaded the folder directly
# !cp -r /content/drive/MyDrive/market-pulse-data/features/* data/features/

# Verify all required files exist
import sys

required_files = [
    "data/features/X_train.npy",
    "data/features/X_val.npy",
    "data/features/X_test.npy",
    "data/features/y_direction_train.npy",
    "data/features/y_direction_val.npy",
    "data/features/y_direction_test.npy",
    "data/features/y_return_train.npy",
    "data/features/y_return_val.npy",
    "data/features/y_return_test.npy",
    "data/features/y_volatility_train.npy",
    "data/features/y_volatility_val.npy",
    "data/features/y_volatility_test.npy",
    "data/features/scaler.pkl",
    "data/features/feature_columns.json",
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print("❌ Missing files:")
    for f in missing:
        print(f"   {f}")
    sys.exit(1)

!ls -lh data/features/
print("\n✅ All feature files verified")

# %%
# Verify GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Sanity-check input shape
import numpy as np
X_train = np.load("data/features/X_train.npy")
y_direction = np.load("data/features/y_direction_train.npy")
print(f"\nX_train shape : {X_train.shape}  (sequences, steps, features)")
print(f"NaN count     : {int(np.isnan(X_train).sum())}")
unique, counts = np.unique(y_direction, return_counts=True)
print(f"y_direction classes: {dict(zip(unique.tolist(), counts.tolist()))}")

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
params['training']['early_stopping_patience'] = 7

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
# - ✅ Best model checkpoint saved to models/{arch}_{task}_best.pt
# - ✅ MLflow experiment logging (per-epoch metrics)
# - ✅ Model registry registration (Production alias)
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

        # Update params for this combination
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        params['training']['model'] = model
        params['training']['task'] = task
        with open('params.yaml', 'w') as f:
            yaml.dump(params, f)

        try:
            result = run_training_pipeline()
            results[f"{model}_{task}"] = result

            model_time = time.time() - model_start
            print(f"\n✅ COMPLETED {model}_{task} in {model_time/60:.1f} minutes")
            print(f"\nMetrics:")
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                elif isinstance(value, (int, str)):
                    print(f"  {metric}: {value}")

            history = result['history']
            if 'train_loss' in history:
                epochs_trained = len(history['train_loss'])
                print(f"\nTraining Summary:")
                print(f"  Epochs: {epochs_trained}/{params['training']['epochs']}")
                print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
                print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
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

    if 'accuracy' in metrics:
        row['Accuracy'] = metrics['accuracy']
    if 'f1_score' in metrics:
        row['F1 Score'] = metrics['f1_score']
    if 'rmse' in metrics:
        row['RMSE'] = metrics['rmse']
    if 'mae' in metrics:
        row['MAE'] = metrics['mae']

    summary.append(row)

df = pd.DataFrame(summary)

print("\n" + "="*100)
print("TRAINING SUMMARY - ALL MODELS")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# Best models per task
print("\n🏆 BEST MODELS PER TASK:")
print("="*100)

for task in ['Direction', 'Volatility']:
    task_df = df[df['Task'] == task.capitalize()]
    if len(task_df) == 0 or 'Accuracy' not in task_df.columns:
        continue
    best = task_df.loc[task_df['Accuracy'].idxmax()]
    print(f"\n{task}: {best['Model']} (Accuracy: {best['Accuracy']:.4f}, F1: {best.get('F1 Score', float('nan')):.4f})")

task_df = df[df['Task'] == 'Return']
if len(task_df) > 0 and 'RMSE' in task_df.columns:
    best = task_df.loc[task_df['RMSE'].idxmin()]
    print(f"\nReturn: {best['Model']} (RMSE: {best['RMSE']:.4f}, MAE: {best.get('MAE', float('nan')):.4f})")

print("\n" + "="*100)

df.to_csv('training_summary.csv', index=False)
print("✅ Summary saved to training_summary.csv")

# %% [markdown]
# ## Verify MLflow Model Registry
#
# Check that all models were registered correctly

# %%
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

        # Check for Production alias (MLflow 2.x) or Production stage (legacy)
        prod_found = False
        for v in versions:
            aliases = getattr(v, 'aliases', [])
            if 'Production' in aliases:
                print(f"   ✅ Production alias: Version {v.version} (run: {v.run_id[:8]}...)")
                prod_found = True
                break
        if not prod_found:
            # Fallback: check legacy stage field
            prod_versions = [v for v in versions if getattr(v, 'current_stage', '') == 'Production']
            if prod_versions:
                prod = prod_versions[0]
                print(f"   ✅ Production stage: Version {prod.version} (run: {prod.run_id[:8]}...)")
            else:
                print(f"   ⚠️  No Production version set")
    except Exception as e:
        print(f"\n❌ {model_name}: {e}")

print("\n" + "="*80)

# %% [markdown]
# ## Package and Download All Artifacts
#
# Creates `trained_models.zip` containing:
# - All 9 trained model checkpoints (`models/*.pt`)
# - `scaler.pkl`  ← **required for inference** (must match training transform)
# - `feature_columns.json` ← feature ordering for the API
# - `training_summary.csv`

# %%
import zipfile
import glob
import shutil

OUTPUT_ZIP = "trained_models.zip"

# Collect all model checkpoint files produced by training
pt_files = glob.glob("models/*.pt")

print(f"Found {len(pt_files)} model checkpoint(s):")
for f in sorted(pt_files):
    size_mb = os.path.getsize(f) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")

if len(pt_files) == 0:
    print("⚠️  No .pt files found in models/ — training may have failed for all models.")

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Model checkpoints
    for pt in pt_files:
        zf.write(pt)

    # Scaler — critical for production inference
    scaler_path = "data/features/scaler.pkl"
    if os.path.exists(scaler_path):
        zf.write(scaler_path, "scaler.pkl")
        print(f"✅ scaler.pkl included")
    else:
        print("❌ scaler.pkl NOT FOUND — models cannot be used for inference without it!")

    # Feature column ordering
    fc_path = "data/features/feature_columns.json"
    if os.path.exists(fc_path):
        zf.write(fc_path, "feature_columns.json")
        print(f"✅ feature_columns.json included")

    # Training summary
    if os.path.exists("training_summary.csv"):
        zf.write("training_summary.csv")
        print(f"✅ training_summary.csv included")

zip_size_mb = os.path.getsize(OUTPUT_ZIP) / 1e6
print(f"\n📦 Archive: {OUTPUT_ZIP}  ({zip_size_mb:.1f} MB)")
print("Contents:")
with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zf:
    for info in zf.infolist():
        print(f"  {info.filename}  ({info.file_size / 1e6:.1f} MB)")

# %%
# Trigger browser download
from google.colab import files
files.download(OUTPUT_ZIP)
print(f"\n✅ Download triggered for {OUTPUT_ZIP}")
print("\n📌 Next steps:")
print("  1. Extract the zip to your local project root")
print("  2. Place models/*.pt → models/")
print("  3. Place scaler.pkl → data/features/scaler.pkl")
print("  4. Place feature_columns.json → data/features/feature_columns.json")
print("  5. All experiments are logged to your Railway MLflow server")
print("  6. Models are registered under MarketPulsePredictor-{task} with alias 'Production'")
