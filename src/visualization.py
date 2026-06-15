from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_RAW_PATH, REPORTS_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "imgs"


def ensure_images_dir():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def save_eda_plots():
    ensure_images_dir()
    df = pd.read_csv(DATA_RAW_PATH)
    sns.set_theme(style="whitegrid")
    log_price = np.log1p(df["Price (INR)"])
    plot_df = df.copy()
    plot_df["LogPrice"] = log_price

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(log_price, bins=50, kde=True, color="teal", ax=axes[0])
    axes[0].set_title("Log-Transformed Price Distribution")
    axes[0].set_xlabel("Log(Price + 1)")
    axes[0].set_ylabel("Count")

    sns.boxplot(data=plot_df, x="Gender", y="LogPrice", ax=axes[1])
    axes[1].set_title("Log(Price) Distribution by Gender")
    axes[1].set_xlabel("Gender Group")
    axes[1].set_ylabel("Price (INR)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "eda_price_plots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(log_price, bins=50, kde=True, color="teal", ax=ax)
    ax.set_title("Log-Transformed Price Distribution")
    ax.set_xlabel("Log(Price + 1)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "log_price_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=plot_df, x="Gender", y="LogPrice", ax=ax)
    ax.set_title("Log(Price) Distribution by Gender")
    ax.set_xlabel("Gender Group")
    ax.set_ylabel("Price (INR)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "log_price_by_gender.png", dpi=200)
    plt.close(fig)


def save_model_comparison_plots():
    ensure_images_dir()
    metrics = pd.read_csv(REPORTS_DIR / "test_metrics.csv")
    sns.set_theme(style="whitegrid")

    baseline = metrics[metrics["model"].isin(["KNN", "SVM", "RandomForest"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=baseline, x="model", y="R2", hue="model", palette="viridis", legend=False, ax=ax)
    ax.set_title("Baseline Models Comparison (R2 Score)")
    ax.set_xlabel("")
    ax.set_ylabel("R2 Score")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "baseline_models_r2.png", dpi=200)
    plt.close(fig)

    comparison = pd.DataFrame({
        "model": ["Best Baseline (RandomForest)", "Deep Learning (MLP)"],
        "R2": [
            metrics.loc[metrics["model"] == "RandomForest", "R2"].iloc[0],
            metrics.loc[metrics["model"] == "MLP", "R2"].iloc[0],
        ],
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=comparison,
        x="model",
        y="R2",
        hue="model",
        palette=["#9bb7eb", "#efb293"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Baseline vs Deep Learning Extension")
    ax.set_xlabel("")
    ax.set_ylabel("R2 Score")
    ax.set_ylim(0, 1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "baseline_vs_deep_learning.png", dpi=200)
    plt.close(fig)


def save_report_plots():
    save_eda_plots()
    save_model_comparison_plots()
    print(f"Report plots saved to {IMAGES_DIR}")


if __name__ == "__main__":
    save_report_plots()
