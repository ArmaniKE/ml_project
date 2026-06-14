import csv
from pathlib import Path
from typing import Dict, Any


def rank_models(cv_results: Dict[str, Dict[str, float]], test_results: Dict[str, Dict[str, float]]):
    ranked = []
    for name, metrics in test_results.items():
        score = metrics.get("R2", 0) - metrics.get("RMSE", 0) * 0.01
        ranked.append({"model": name, "score": score, **metrics})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def export_metrics_csv(results: Dict[str, Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["model"] + sorted({k for metrics in results.values() for k in metrics.keys()})
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            row = {"model": model_name, **metrics}
            writer.writerow(row)


def create_summary_report(cv_results: Dict[str, Dict[str, float]], test_results: Dict[str, Dict[str, float]]):
    ranking = rank_models(cv_results, test_results)
    return {
        "summary": {
            "best_model": ranking[0]["model"] if ranking else None,
            "model_count": len(ranking),
        },
        "ranking": ranking,
    }


def create_enterprise_comparison_report(cv_results, test_results, dl_history=None):
    summary = create_summary_report(cv_results, test_results)
    summary["cross_validation"] = cv_results
    summary["test_metrics"] = test_results
    if dl_history is not None:
        summary["deep_learning_history"] = dl_history
    return summary
