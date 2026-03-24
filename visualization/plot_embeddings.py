from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from word2vec.artifacts import embedding_matrix, load_artifact, nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Word2Vec embeddings in 2D.")
    parser.add_argument("--models-dir", default="output/models", help="Directory containing trained artifacts.")
    parser.add_argument("--model-paths", nargs="*", help="Optional explicit model artifact paths.")
    parser.add_argument("--plots-dir", default="output/plots", help="Directory to save figures.")
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne", help="Projection method.")
    parser.add_argument("--max-words", type=int, default=120, help="Maximum words to include in the plot.")
    return parser.parse_args()


def select_default_model_paths(models_dir: str) -> list[str]:
    summary_path = os.path.join(models_dir, "experiment_summary.json")
    if os.path.exists(summary_path):
        try:
            import json

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
    return sorted(glob.glob(os.path.join(models_dir, "*.pt")))[:2]


def build_word_list(artifact: dict, max_words: int) -> list[str]:
    focus_words = [
        "research",
        "student",
        "phd",
        "exam",
        "course",
        "faculty",
        "btech",
        "mtech",
        "ug",
        "pg",
        "department",
        "academic",
    ]
    counts = artifact["vocab"]["counts"]
    tokens = artifact["vocab"]["itos"]
    selected = [word for word in focus_words if word in tokens]

    for word in ("research", "student", "academic"):
        for neighbor, _ in nearest_neighbors(artifact, word, topn=10):
            if neighbor not in selected:
                selected.append(neighbor)

    for token, _count in sorted(zip(tokens, counts), key=lambda item: item[1], reverse=True):
        if token not in selected:
            selected.append(token)
        if len(selected) >= max_words:
            break

    return selected[:max_words]


def project_vectors(vectors: np.ndarray, method: str) -> np.ndarray:
    if len(vectors) < 2:
        raise ValueError("Need at least two vectors for 2D projection.")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(vectors)

    pca_dims = min(50, vectors.shape[1], len(vectors))
    reduced = PCA(n_components=pca_dims, random_state=42).fit_transform(vectors)
    perplexity = min(30, max(5, len(vectors) // 4))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    return reducer.fit_transform(reduced)


def plot_embeddings(model_path: str, output_dir: str, method: str, max_words: int) -> None:
    artifact = load_artifact(model_path)
    model_type = artifact["model_type"]
    config = artifact["config"]
    title = f"{model_type.upper()} ({method.upper()}, dim={config['embedding_dim']}, win={config['window_size']})"

    print(f"Loading {title}...")
    words_to_plot = build_word_list(artifact, max_words=max_words)
    stoi = {token: idx for idx, token in enumerate(artifact["vocab"]["itos"])}
    vectors = embedding_matrix(artifact).numpy()
    word_vectors = np.stack([vectors[stoi[word]] for word in words_to_plot], axis=0)

    print(f"Projecting {len(words_to_plot)} words using {method.upper()}...")
    projected = project_vectors(word_vectors, method)

    focus_words = {"research", "student", "phd", "exam", "course", "faculty", "btech", "mtech", "ug", "pg"}
    plt.figure(figsize=(14, 10))
    for index, word in enumerate(words_to_plot):
        x_coord, y_coord = projected[index]
        if word in focus_words:
            plt.scatter(x_coord, y_coord, color="crimson", s=60, edgecolors="black")
            plt.annotate(
                word,
                (x_coord, y_coord),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="maroon",
            )
        else:
            plt.scatter(x_coord, y_coord, color="steelblue", s=20, alpha=0.55)
            if index % 12 == 0:
                plt.annotate(
                    word,
                    (x_coord, y_coord),
                    xytext=(4, 2),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.75,
                )

    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_{method}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_file}")


def main():
    args = parse_args()
    model_paths = args.model_paths or select_default_model_paths(args.models_dir)
    if not model_paths:
        print(f"No model artifacts found under {args.models_dir}. Train models first.")
        return

    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Skipping missing artifact: {model_path}")
            continue
        plot_embeddings(model_path, args.plots_dir, args.method, args.max_words)

    print("\nVisualization step complete.")
    print("Interpretation tip: CBOW often yields smoother local clusters, while Skip-gram can separate rarer semantics more clearly.")


if __name__ == "__main__":
    main()
