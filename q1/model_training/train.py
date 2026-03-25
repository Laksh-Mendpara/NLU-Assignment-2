from __future__ import annotations

"""Run the Q1 Word2Vec experiment grid and save all outputs.

I kept the comments simple so the training flow is easy to follow.
"""

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
    # I fixed the experiment setup here so the script stays simple.
    return argparse.Namespace(
        corpus_file="output/processed_corpus.txt",
        out_dir="output/models",
        architectures=["cbow", "skipgram"],
        dimensions=[100, 200, 300],
        windows=[2, 5],
        negatives=[5, 10],
        epochs=120,
        batch_size=128,
        learning_rate=0.005,
        min_count=3,
        subsample_threshold=1e-5,
        disable_dynamic_window=False,
        seed=42,
        validation_ratio=0.05,
        early_stopping_patience=8,
        early_stopping_min_delta=0.0005,
        early_stopping_min_epochs=20,
        lr_patience=3,
        lr_decay_factor=0.5,
        min_learning_rate=0.0001,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def set_seed(seed: int) -> None:
    # A fixed seed makes the train/validation split and training runs repeatable.
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
        "retained_token_ratio_per_epoch",
        "lr_history",
        "best_validation_loss",
        "best_epoch",
        "epochs_completed",
        "stopped_early",
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

    rng = random.Random(args.seed)
    sentence_indices = list(range(len(encoded_sentences)))
    rng.shuffle(sentence_indices)
    validation_count = 0
    # Keep one fixed validation split so all experiment settings are compared fairly.
    if args.validation_ratio > 0:
        # Formula:
        # validation_count = round(total_sentences * validation_ratio)
        validation_count = max(1, int(round(len(sentence_indices) * args.validation_ratio)))
        validation_count = min(validation_count, max(len(sentence_indices) - 1, 0))
    validation_indices = set(sentence_indices[:validation_count])
    train_encoded_sentences = [
        sentence for index, sentence in enumerate(encoded_sentences) if index not in validation_indices
    ]
    validation_encoded_sentences = [
        sentence for index, sentence in enumerate(encoded_sentences) if index in validation_indices
    ]
    if not train_encoded_sentences:
        train_encoded_sentences = encoded_sentences
        validation_encoded_sentences = []

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print("STARTING PYTORCH WORD2VEC EXPERIMENTS")
    print("=" * 60)
    print(f"Vocabulary size: {vocabulary.size}")
    print(f"Training device: {args.device}")
    print(f"Training sentences: {len(train_encoded_sentences)}")
    print(f"Validation sentences: {len(validation_encoded_sentences)}")
    if args.subsample_threshold > 0:
        print("Subsampling: only enabled when the cleaned corpus is large enough to preserve most tokens.")

    summary_records: list[dict] = []
    # Train one model for every config in the small experiment grid.
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
                        encoded_sentences=train_encoded_sentences,
                        vocabulary=vocabulary,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        device=args.device,
                        seed=args.seed,
                        validation_encoded_sentences=validation_encoded_sentences,
                        subsample_threshold=args.subsample_threshold,
                        dynamic_window=not args.disable_dynamic_window,
                        lr_patience=args.lr_patience,
                        lr_decay_factor=args.lr_decay_factor,
                        min_learning_rate=args.min_learning_rate,
                        early_stopping_patience=args.early_stopping_patience,
                        early_stopping_min_delta=args.early_stopping_min_delta,
                        early_stopping_min_epochs=args.early_stopping_min_epochs,
                        verbose=True,
                        log_prefix="    ",
                    )
                    save_artifact(artifact, artifact_path)

                    duration = time.time() - start_time
                    final_loss = artifact["training"]["final_loss"]
                    effective_subsample_threshold = artifact["config"]["subsample_threshold"]
                    subsampling_disabled_reason = artifact["training"]["subsampling_disabled_reason"]
                    retained_tokens = artifact["training"]["estimated_retained_tokens_per_epoch"]
                    retained_ratio = artifact["training"]["retained_token_ratio_per_epoch"]
                    examples_per_epoch = artifact["training"]["examples_per_epoch"]
                    best_validation_loss = artifact["training"]["best_validation_loss"]
                    best_epoch = artifact["training"]["best_epoch"]
                    epochs_completed = artifact["training"]["epochs_completed"]
                    stopped_early = artifact["training"]["stopped_early"]
                    lr_history = artifact["training"].get("lr_history", [])
                    print(f"  -> saved to {artifact_path}")
                    print(f"  -> final average loss: {final_loss:.4f}")
                    if best_epoch:
                        print(
                            f"  -> best validation loss: {best_validation_loss:.4f} "
                            f"(epoch {best_epoch}/{epochs_completed})"
                        )
                    # `retained_ratio` means:
                    # retained_tokens_after_subsampling / total_tokens_in_corpus
                    print(f"  -> retained tokens/epoch: {retained_tokens:.0f} ({retained_ratio:.1%} of corpus)")
                    print(f"  -> examples/epoch: {examples_per_epoch:,}")
                    print(f"  -> early stopping: {'triggered' if stopped_early else 'not triggered'}")
                    if lr_history:
                        print(f"  -> learning-rate schedule: {', '.join(f'{value:.5f}' for value in lr_history)}")
                    if subsampling_disabled_reason:
                        print(f"  -> subsampling: {subsampling_disabled_reason}")
                    else:
                        print(f"  -> subsampling threshold used: {effective_subsample_threshold}")
                    print(f"  -> finished in {duration:.2f} seconds")

                    # Save a compact row so later analysis scripts can pick the best runs.
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
                            "retained_token_ratio_per_epoch": artifact["training"]["retained_token_ratio_per_epoch"],
                            "lr_history": " ".join(f"{value:.5f}" for value in lr_history),
                            "best_validation_loss": artifact["training"]["best_validation_loss"],
                            "best_epoch": artifact["training"]["best_epoch"],
                            "epochs_completed": artifact["training"]["epochs_completed"],
                            "stopped_early": artifact["training"]["stopped_early"],
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
