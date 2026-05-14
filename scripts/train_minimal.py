#!/usr/bin/env python3
"""
Minimal training script for quick local testing (CPU-only).
Use this ONLY for testing the pipeline, not for production models.

For production training, use Colab with GPU (see TRAINING_GUIDE.md).
"""

from pathlib import Path

import yaml

# Reduce training time for quick testing
MINIMAL_PARAMS = {
    "training": {
        "epochs": 3,  # Instead of 50
        "batch_size": 16,  # Smaller batches
        "learning_rate": 0.001,
        "hidden_size": 32,  # Much smaller
        "num_layers": 1,  # Single layer
        "dropout": 0.1,
        "early_stopping_patience": 2,
        "device": "cpu",
    }
}


def main():
    print("⚠️  MINIMAL TRAINING MODE")
    print("=" * 60)
    print("This trains tiny models quickly for TESTING ONLY.")
    print("For production models, use Colab GPU training.")
    print("See TRAINING_GUIDE.md for details.")
    print("=" * 60)
    print()

    # Backup original params
    params_path = Path("params.yaml")
    backup_path = Path("params.yaml.backup")

    if params_path.exists():
        params_path.rename(backup_path)
        print(f"✅ Backed up params.yaml to {backup_path}")

    # Load and update params
    with open(backup_path, "r") as f:
        params = yaml.safe_load(f)

    params["training"].update(MINIMAL_PARAMS["training"])

    # Train one model of each type for testing
    models = ["lstm"]  # Just LSTM for speed
    tasks = ["direction"]  # Just one task

    from src.training.run_training import run_training_pipeline

    for model in models:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Training {model.upper()} for {task} task (minimal)")
            print(f"{'='*60}\n")

            params["training"]["model"] = model
            params["training"]["task"] = task

            with open(params_path, "w") as f:
                yaml.dump(params, f)

            try:
                result = run_training_pipeline()
                print(f"\n✅ Completed {model}_{task}")
                print(f"Metrics: {result['metrics']}")
            except Exception as e:
                print(f"\n❌ Failed {model}_{task}: {e}")

    # Restore original params
    if backup_path.exists():
        backup_path.rename(params_path)
        print("\n✅ Restored original params.yaml")

    print("\n" + "=" * 60)
    print("MINIMAL TRAINING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check models/ directory for trained model")
    print("2. Test API: uvicorn src.api.main:app --reload")
    print("3. For production models, use Colab GPU training")


if __name__ == "__main__":
    main()
