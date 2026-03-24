from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from word2vec.artifacts import load_artifact, nearest_neighbors, solve_analogy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic analysis on trained PyTorch Word2Vec models.")
    parser.add_argument("--models-dir", default="output/models", help="Directory containing .pt model artifacts.")
    parser.add_argument("--model-paths", nargs="*", help="Optional explicit model artifact paths.")
    parser.add_argument("--topn", type=int, default=5, help="Number of nearest neighbors to report.")
    parser.add_argument("--analogy-topn", type=int, default=5, help="Number of analogy candidates to report.")
    parser.add_argument(
        "--embedding-mode",
        choices=["combined", "input", "output"],
        default="combined",
        help="Embedding matrix to use for similarity and analogy analysis.",
    )
    return parser.parse_args()


def select_default_model_paths(models_dir: str) -> list[str]:
    """Pick the best CBOW artifact and best Skip-gram artifact for reporting."""
    summary_path = os.path.join(models_dir, "experiment_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            selected: list[str] = []
            for architecture in ("cbow", "skipgram"):
                candidates = [record for record in records if record.get("model_type") == architecture]
                if not candidates:
                    continue
                best = min(candidates, key=lambda record: record.get("final_loss", float("inf")))
                artifact_path = best.get("artifact_path")
                if artifact_path and os.path.exists(artifact_path):
                    selected.append(artifact_path)
            if selected:
                return selected
        except Exception:
            pass

    preferred = [
        os.path.join(models_dir, "cbow_dim100_win2_neg5.pt"),
        os.path.join(models_dir, "skipgram_dim100_win2_neg5.pt"),
        os.path.join(models_dir, "cbow_dim100_win5_neg5.pt"),
        os.path.join(models_dir, "skipgram_dim100_win5_neg5.pt"),
    ]
    discovered = [path for path in preferred if os.path.exists(path)]
    if discovered:
        return discovered[:2]

    paths = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    selected: list[str] = []
    for architecture in ("cbow", "skipgram"):
        for path in paths:
            if os.path.basename(path).startswith(f"{architecture}_"):
                selected.append(path)
                break
    return selected


def run_semantic_analysis(model_path: str, topn: int, analogy_topn: int, embedding_mode: str) -> dict:
    artifact = load_artifact(model_path)
    model_type = artifact["model_type"]
    config = artifact["config"]
    label = (
        f"{model_type.upper()} "
        f"(dim={config['embedding_dim']}, win={config['window_size']}, neg={config['negative_samples']})"
    )

    print("\n" + "=" * 60)
    print(f" SEMANTIC ANALYSIS: {label}")
    print("=" * 60)

    target_words = ["research", "student", "phd", "exam"]
    analysis = {"model_path": model_path, "model_label": label, "neighbors": {}, "analogies": {}}

    print(f"\n--- 1. Top {topn} Nearest Neighbors ---")
    for word in target_words:
        neighbors = nearest_neighbors(artifact, word, topn=topn, mode=embedding_mode)
        if not neighbors:
            print(f"\nWord '{word}' not in vocabulary.")
            analysis["neighbors"][word] = []
            continue

        print(f"\nNeighbors for '{word}':")
        for neighbor, similarity in neighbors:
            print(f"  - {neighbor} (cos: {similarity:.4f})")
        analysis["neighbors"][word] = neighbors

    analogies = [
        ("ug", "btech", "pg"),
        ("department", "faculty", "student"),
        ("course", "exam", "research"),
        ("btech", "undergraduate", "mtech"),
    ]
    print("\n--- 2. Analogy Experiments ---")
    for a, b, c in analogies:
        predictions = solve_analogy(artifact, a, b, c, topn=analogy_topn, mode=embedding_mode)
        key = f"{a}:{b}::{c}:?"
        if not predictions:
            print(f"\nSkipping analogy '{a}' : '{b}' :: '{c}' : ? (missing vocabulary items)")
            analysis["analogies"][key] = []
            continue

        print(f"\nAnalogy: '{a}' is to '{b}' as '{c}' is to ?")
        for word, similarity in predictions:
            print(f"  -> {word} (cos: {similarity:.4f})")
        analysis["analogies"][key] = predictions

    return analysis


def main():
    args = parse_args()
    model_paths = args.model_paths or select_default_model_paths(args.models_dir)
    if not model_paths:
        print(f"No model artifacts found under {args.models_dir}. Train models first.")
        return

    report = []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Skipping missing artifact: {model_path}")
            continue
        report.append(
            run_semantic_analysis(
                model_path,
                topn=args.topn,
                analogy_topn=args.analogy_topn,
                embedding_mode=args.embedding_mode,
            )
        )

    report_dir = args.models_dir
    if args.model_paths:
        report_dir = os.path.dirname(os.path.abspath(model_paths[0])) or args.models_dir
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "semantic_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSemantic analysis report saved to {report_path}")


if __name__ == "__main__":
    main()
