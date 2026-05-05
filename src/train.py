import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

try:
    from .data_loader import load_image_dataset
    from .evaluate import evaluate_model, save_classification_outputs
    from .models import build_model
except ImportError:
    from data_loader import load_image_dataset
    from evaluate import evaluate_model, save_classification_outputs
    from models import build_model


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def save_history(history, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path / "learning_curves.png", dpi=200)
    plt.close(fig)


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, info = load_image_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.model, input_shape=info.input_shape, num_classes=info.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = Path(args.output_dir) / info.name.lower().replace("-", "_") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "resolved_dataset_name": info.name,
            "input_shape": info.input_shape,
            "num_classes": info.num_classes,
            "device": str(device),
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    history = []
    best_val_accuracy = 0.0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f}"
        )

        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_results = evaluate_model(model, train_loader, criterion, device)
    val_results = evaluate_model(model, val_loader, criterion, device)
    test_results = evaluate_model(model, test_loader, criterion, device)

    summary = {
        "dataset": info.name,
        "model": args.model,
        "train_loss": train_results["loss"],
        "train_accuracy": train_results["accuracy"],
        "val_loss": val_results["loss"],
        "val_accuracy": val_results["accuracy"],
        "test_loss": test_results["loss"],
        "test_accuracy": test_results["accuracy"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_history(history, output_dir)
    report = save_classification_outputs(test_results, info.class_names, output_dir)

    print("\nFinal summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"weighted_f1: {report['weighted avg']['f1-score']}")
    print(f"Outputs saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "cifar10"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "simple_cnn", "lenet", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
