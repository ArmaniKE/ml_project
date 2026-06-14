import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.pipeline import run_training_pipeline

def main():
    print("Starting the ML pipeline execution")
    trained_models, preprocessor, report = run_training_pipeline()
    print("Training pipeline complete.")
    print("Report summary:")
    print(report)

if __name__ == "__main__":
    main()
