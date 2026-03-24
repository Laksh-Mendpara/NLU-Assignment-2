from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from word2vec.artifacts import save_artifact
from word2vec.training import (
    build_vocabulary,
    encode_corpus,
    load_tokenized_corpus,
    train_word2vec_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CBOW and Skip-gram from scratch in PyTorch.")
    parser.add_argument("--corpus-file", default="output/processed_corpus.txt", help="Tokenized corpus file.")
    parser.add_argument("--out-dir", default="output/models", help="Directory for model artifacts.")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["cbow", "skipgram"],
        choices=["cbow", "skipgram"],
        help="Word2Vec architectures to train.",
    )
    parser.add_argument("--dimensions", nargs="+", type=int, default=[50, 100], help="Embedding dimensions.")
    parser.add_argument("--windows", nargs="+", type=int, default=[2, 5], help="Context window sizes.")
    parser.add_argument("--negatives", nargs="+", type=int, default=[5, 10], help="Negative samples.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for each experiment.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="SparseAdam learning rate.")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum frequency threshold for vocabulary.")
    parser.add_argument(
        "--subsample-threshold",
        type=float,
        default=1e-5,
        help="Frequent-word subsampling threshold. It is auto-disabled on small corpora.",
    )
    parser.add_argument(
        "--disable-dynamic-window",
        action="store_true",
        help="Use a fixed context window instead of randomizing up to the max window.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device: cpu or cuda.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_experiment_summaries(records: list[dict], out_dir: str) -> None:
    """Persist experiment metrics in JSON and CSV formats."""
    json_path = os.path.join(out_dir, "experiment_summary.json")
    csv_path = os.path.join(out_dir, "experiment_summary.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    fieldnames = [
        "model_type",
        "embedding_dim",
        "window_size",
        "negative_samples",
        "epochs",
        "batch_size",
        "vocab_size",
        "total_tokens",
        "examples_per_epoch",
        "estimated_retained_tokens_per_epoch",
        "final_loss",
        "artifact_path",
        "training_seconds",
        "subsample_threshold",
        "dynamic_window",
        "subsampling_disabled_reason",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    if not os.path.exists(args.corpus_file):
        print(f"Corpus not found at {args.corpus_file}. Run preprocessing first.")
        return

    set_seed(args.seed)

    print("Loading tokenized corpus...")
    sentences = load_tokenized_corpus(args.corpus_file)
    print(f"Loaded {len(sentences)} documents.")

    vocabulary = build_vocabulary(sentences, min_count=args.min_count)
    encoded_sentences = encode_corpus(sentences, vocabulary)
    if vocabulary.size == 0 or not encoded_sentences:
        print("Vocabulary is empty after min_count filtering. Lower --min-count or inspect preprocessing output.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print("STARTING PYTORCH WORD2VEC EXPERIMENTS")
    print("=" * 60)
    print(f"Vocabulary size: {vocabulary.size}")
    print(f"Training device: {args.device}")
    if args.subsample_threshold > 0:
        print("Subsampling: auto-disabled on small corpora to avoid collapsing training pairs.")

    summary_records: list[dict] = []
    for architecture in args.architectures:
        for dimension in args.dimensions:
            for window_size in args.windows:
                for negative_samples in args.negatives:
                    model_name = f"{architecture}_dim{dimension}_win{window_size}_neg{negative_samples}"
                    artifact_path = os.path.join(args.out_dir, f"{model_name}.pt")
                    print(f"\nTraining {model_name}...")
                    start_time = time.time()

                    artifact = train_word2vec_model(
                        model_type=architecture,
                        embedding_dim=dimension,
                        window_size=window_size,
                        negative_samples=negative_samples,
                        encoded_sentences=encoded_sentences,
                        vocabulary=vocabulary,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        device=args.device,
                        seed=args.seed,
                        subsample_threshold=args.subsample_threshold,
                        dynamic_window=not args.disable_dynamic_window,
                    )
                    save_artifact(artifact, artifact_path)

                    duration = time.time() - start_time
                    final_loss = artifact["training"]["final_loss"]
                    effective_subsample_threshold = artifact["config"]["subsample_threshold"]
                    subsampling_disabled_reason = artifact["training"]["subsampling_disabled_reason"]
                    print(f"  -> saved to {artifact_path}")
                    print(f"  -> final average loss: {final_loss:.4f}")
                    if subsampling_disabled_reason:
                        print(f"  -> subsampling: {subsampling_disabled_reason}")
                    else:
                        print(f"  -> subsampling threshold used: {effective_subsample_threshold}")
                    print(f"  -> finished in {duration:.2f} seconds")

                    summary_records.append(
                        {
                            "model_type": architecture,
                            "embedding_dim": dimension,
                            "window_size": window_size,
                            "negative_samples": negative_samples,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "vocab_size": artifact["training"]["vocab_size"],
                            "total_tokens": artifact["training"]["total_tokens"],
                            "examples_per_epoch": artifact["training"]["examples_per_epoch"],
                            "estimated_retained_tokens_per_epoch": artifact["training"]["estimated_retained_tokens_per_epoch"],
                            "final_loss": final_loss,
                            "artifact_path": artifact_path,
                            "training_seconds": round(duration, 2),
                            "subsample_threshold": effective_subsample_threshold,
                            "dynamic_window": not args.disable_dynamic_window,
                            "subsampling_disabled_reason": subsampling_disabled_reason,
                        }
                    )

    write_experiment_summaries(summary_records, args.out_dir)
    print("\nAll models trained and saved successfully.")
    print(f"Experiment summaries written to {args.out_dir}")


if __name__ == "__main__":
    main()
