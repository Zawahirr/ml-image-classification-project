import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "targets": np.array(all_targets),
        "predictions": np.array(all_predictions),
    }


def save_classification_outputs(results, class_names, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    targets = results["targets"]
    predictions = results["predictions"]

    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with (output_path / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(targets, predictions)
    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    return report
