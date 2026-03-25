from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def save_artifact(artifact: dict, path: str | Path) -> None:
    """Persist a trained model artifact to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_artifact(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a saved model artifact."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def token_to_index(artifact: dict) -> dict[str, int]:
    """Reconstruct the token-to-index lookup from the artifact."""
    return {token: idx for idx, token in enumerate(artifact["vocab"]["itos"])}


def _ranked_results(
    similarities: torch.Tensor,
    tokens: list[str],
    topn: int,
    excluded_indices: set[int] | None = None,
) -> list[tuple[str, float]]:
    # This sorts scores from high to low and keeps the best few words.
    results: list[tuple[str, float]] = []
    for idx in torch.argsort(similarities, descending=True).tolist():
        if excluded_indices and idx in excluded_indices:
            continue
        results.append((tokens[idx], float(similarities[idx].item())))
        if len(results) >= topn:
            break
    return results


def embedding_matrix(artifact: dict, mode: str = "combined") -> torch.Tensor:
    """
    Return the embedding matrix used for downstream analysis.

    `combined` averages input and output embeddings, which usually performs
    better for SGNS-style evaluation than using either matrix alone.
    """
    embeddings = artifact["embeddings"]
    input_weights = embeddings["input"]
    output_weights = embeddings["output"]

    if mode == "input":
        return input_weights
    if mode == "output":
        return output_weights
    if mode == "combined":
        # Formula:
        # combined_embedding = (input_embedding + output_embedding) / 2
        return (input_weights + output_weights) / 2.0
    raise ValueError(f"Unsupported embedding mode: {mode}")


def nearest_neighbors(
    artifact: dict,
    word: str,
    topn: int = 5,
    mode: str = "combined",
    exclude_tokens: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Return nearest neighbors of a token using cosine similarity."""
    stoi = token_to_index(artifact)
    if word not in stoi:
        return []

    vectors = embedding_matrix(artifact, mode=mode)
    normalized = F.normalize(vectors, p=2, dim=1)
    query_index = stoi[word]
    # Formula:
    # cosine_similarity(a, b) = normalized_a dot normalized_b
    similarities = normalized @ normalized[query_index]
    tokens = artifact["vocab"]["itos"]
    excluded_indices = {query_index}
    if exclude_tokens:
        excluded_indices.update(stoi[token] for token in exclude_tokens if token in stoi)
    return _ranked_results(similarities, tokens, topn=topn, excluded_indices=excluded_indices)


def solve_analogy(
    artifact: dict,
    a: str,
    b: str,
    c: str,
    topn: int = 3,
    mode: str = "combined",
    exclude_tokens: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Solve analogies of the form A:B :: C:? using cosine similarity."""
    stoi = token_to_index(artifact)
    if any(word not in stoi for word in (a, b, c)):
        return []

    vectors = embedding_matrix(artifact, mode=mode)
    normalized = F.normalize(vectors, p=2, dim=1)
    # Formula:
    # analogy_vector = vec(b) - vec(a) + vec(c)
    query = normalized[stoi[b]] - normalized[stoi[a]] + normalized[stoi[c]]
    query = F.normalize(query.unsqueeze(0), p=2, dim=1).squeeze(0)

    similarities = normalized @ query
    tokens = artifact["vocab"]["itos"]
    excluded_indices = {stoi[word] for word in (a, b, c)}
    if exclude_tokens:
        excluded_indices.update(stoi[token] for token in exclude_tokens if token in stoi)
    return _ranked_results(similarities, tokens, topn=topn, excluded_indices=excluded_indices)
