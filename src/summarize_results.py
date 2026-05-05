import argparse
import csv
import json
from pathlib import Path


def collect_summaries(output_dir):
    rows = []
    for summary_path in Path(output_dir).glob("*/*/summary.json"):
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        report_path = summary_path.parent / "classification_report.json"
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as file:
                report = json.load(file)
            summary["weighted_precision"] = report["weighted avg"]["precision"]
            summary["weighted_recall"] = report["weighted avg"]["recall"]
            summary["weighted_f1"] = report["weighted avg"]["f1-score"]

        rows.append(summary)

    return sorted(rows, key=lambda row: (row["dataset"], row["model"]))


def save_summary_table(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No summary.json files found.")

    columns = [
        "dataset",
        "model",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "train_loss",
        "val_loss",
        "test_loss",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a CSV table from completed experiment summaries.")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--table-path", default="./outputs/comparison_table.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment_rows = collect_summaries(args.output_dir)
    save_summary_table(experiment_rows, args.table_path)
    print(f"Saved {len(experiment_rows)} rows to {args.table_path}")
