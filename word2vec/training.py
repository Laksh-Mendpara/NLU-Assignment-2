from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]
    counts: list[int]

    @property
    def size(self) -> int:
        return len(self.itos)


def load_tokenized_corpus(file_path: str) -> list[list[str]]:
    """Load a whitespace-tokenized corpus where each line is one document."""
    sentences: list[list[str]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def build_vocabulary(sentences: list[list[str]], min_count: int) -> Vocabulary:
    """Build a frequency-filtered vocabulary."""
    counter = Counter(token for sentence in sentences for token in sentence)
    items = sorted(
        ((token, count) for token, count in counter.items() if count >= min_count),
        key=lambda item: (-item[1], item[0]),
    )

    stoi = {token: idx for idx, (token, _) in enumerate(items)}
    itos = [token for token, _ in items]
    counts = [count for _, count in items]
    return Vocabulary(stoi=stoi, itos=itos, counts=counts)


def encode_corpus(sentences: list[list[str]], vocabulary: Vocabulary) -> list[list[int]]:
    """Map the tokenized corpus to vocabulary indices and drop OOV tokens."""
    encoded: list[list[int]] = []
    for sentence in sentences:
        ids = [vocabulary.stoi[token] for token in sentence if token in vocabulary.stoi]
        if len(ids) >= 2:
            encoded.append(ids)
    return encoded


def count_training_examples(encoded_sentences: list[list[int]], window_size: int, model_type: str) -> int:
    """Count the number of examples produced by the corpus."""
    total = 0
    for sentence in encoded_sentences:
        for position in range(len(sentence)):
            left = max(0, position - window_size)
            right = min(len(sentence), position + window_size + 1)
            context_len = (position - left) + (right - position - 1)
            if context_len <= 0:
                continue
            total += context_len if model_type == "skipgram" else 1
    return total


def build_subsampling_keep_probabilities(
    counts: list[int],
    total_tokens: int,
    threshold: float,
) -> list[float]:
    """Compute Mikolov-style keep probabilities for frequent-token subsampling."""
    if threshold <= 0 or total_tokens <= 0:
        return [1.0 for _ in counts]

    keep_probabilities: list[float] = []
    for count in counts:
        frequency = count / total_tokens
        keep_probability = min(1.0, math.sqrt(threshold / frequency) + (threshold / frequency))
        keep_probabilities.append(keep_probability)
    return keep_probabilities


def expected_retained_tokens(
    encoded_sentences: list[list[int]],
    keep_probabilities: list[float] | None,
) -> float:
    """Estimate how many tokens remain after subsampling."""
    if keep_probabilities is None:
        return float(sum(len(sentence) for sentence in encoded_sentences))
    return float(sum(sum(keep_probabilities[token] for token in sentence) for sentence in encoded_sentences))


def subsample_sentence(
    sentence: list[int],
    keep_probabilities: list[float] | None,
    rng: random.Random,
) -> list[int]:
    """Drop extremely frequent words before generating training pairs."""
    if keep_probabilities is None:
        return sentence
    filtered = [token for token in sentence if rng.random() <= keep_probabilities[token]]
    return filtered if len(filtered) >= 2 else []


class NegativeSampler:
    """Sample negatives from the unigram^0.75 distribution."""

    def __init__(self, counts: list[int], exponent: float = 0.75, device: str | torch.device = "cpu") -> None:
        noise = torch.tensor(counts, dtype=torch.float32, device=device)
        noise = noise.pow(exponent)
        self.probabilities = noise / noise.sum()
        self.device = device

    def sample(self, batch_size: int, num_samples: int, forbidden: torch.Tensor | None = None) -> torch.Tensor:
        samples = torch.multinomial(
            self.probabilities,
            batch_size * num_samples,
            replacement=True,
        ).view(batch_size, num_samples)

        if forbidden is None:
            return samples

        forbidden = forbidden.to(self.device)
        if forbidden.dim() == 1:
            forbidden = forbidden.unsqueeze(1)

        collision_mask = (samples.unsqueeze(-1) == forbidden.unsqueeze(1)).any(dim=-1)
        while collision_mask.any():
            replacement = torch.multinomial(
                self.probabilities,
                int(collision_mask.sum().item()),
                replacement=True,
            )
            samples[collision_mask] = replacement
            collision_mask = (samples.unsqueeze(-1) == forbidden.unsqueeze(1)).any(dim=-1)

        return samples


class CBOWNegativeSamplingModel(nn.Module):
    """CBOW model trained with negative sampling."""

    def __init__(self, vocab_size: int, embedding_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=pad_idx, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.pad_idx = pad_idx
        self._reset_parameters(embedding_dim)

    def _reset_parameters(self, embedding_dim: int) -> None:
        bound = 0.5 / max(embedding_dim, 1)
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)
        with torch.no_grad():
            self.input_embeddings.weight[self.pad_idx].fill_(0.0)

    def forward(
        self,
        contexts: torch.Tensor,
        context_mask: torch.Tensor,
        targets: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        context_vectors = self.input_embeddings(contexts)
        masked_vectors = context_vectors * context_mask.unsqueeze(-1)
        pooled = masked_vectors.sum(dim=1) / context_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        positive_vectors = self.output_embeddings(targets)
        negative_vectors = self.output_embeddings(negatives)

        positive_scores = (pooled * positive_vectors).sum(dim=1)
        negative_scores = torch.einsum("bd,bkd->bk", pooled, negative_vectors)

        positive_loss = F.logsigmoid(positive_scores)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        return -(positive_loss + negative_loss).mean()


class SkipGramNegativeSamplingModel(nn.Module):
    """Skip-gram model trained with negative sampling."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self._reset_parameters(embedding_dim)

    def _reset_parameters(self, embedding_dim: int) -> None:
        bound = 0.5 / max(embedding_dim, 1)
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        centers: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        center_vectors = self.input_embeddings(centers)
        positive_vectors = self.output_embeddings(positives)
        negative_vectors = self.output_embeddings(negatives)

        positive_scores = (center_vectors * positive_vectors).sum(dim=1)
        negative_scores = torch.einsum("bd,bkd->bk", center_vectors, negative_vectors)

        positive_loss = F.logsigmoid(positive_scores)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        return -(positive_loss + negative_loss).mean()


def _collate_cbow(
    contexts_batch: list[list[int]],
    targets_batch: list[int],
    pad_idx: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_context_length = max(len(context) for context in contexts_batch)
    contexts = torch.full(
        (len(contexts_batch), max_context_length),
        fill_value=pad_idx,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros((len(contexts_batch), max_context_length), dtype=torch.float32, device=device)

    for row, context in enumerate(contexts_batch):
        contexts[row, : len(context)] = torch.tensor(context, dtype=torch.long, device=device)
        mask[row, : len(context)] = 1.0

    targets = torch.tensor(targets_batch, dtype=torch.long, device=device)
    return contexts, mask, targets


def generate_cbow_batches(
    encoded_sentences: list[list[int]],
    window_size: int,
    batch_size: int,
    pad_idx: int,
    device: str | torch.device,
    rng: random.Random,
    keep_probabilities: list[float] | None = None,
    dynamic_window: bool = True,
):
    """Yield CBOW mini-batches without materializing all examples in memory."""
    sentence_indices = list(range(len(encoded_sentences)))
    rng.shuffle(sentence_indices)

    contexts_batch: list[list[int]] = []
    targets_batch: list[int] = []

    for sentence_index in sentence_indices:
        sentence = subsample_sentence(encoded_sentences[sentence_index], keep_probabilities, rng)
        if len(sentence) < 2:
            continue
        for position, target in enumerate(sentence):
            current_window = rng.randint(1, window_size) if dynamic_window else window_size
            left = max(0, position - current_window)
            right = min(len(sentence), position + current_window + 1)
            context = sentence[left:position] + sentence[position + 1:right]
            if not context:
                continue

            contexts_batch.append(context)
            targets_batch.append(target)

            if len(targets_batch) == batch_size:
                yield _collate_cbow(contexts_batch, targets_batch, pad_idx, device)
                contexts_batch, targets_batch = [], []

    if targets_batch:
        yield _collate_cbow(contexts_batch, targets_batch, pad_idx, device)


def _collate_skipgram(
    centers_batch: list[int],
    positives_batch: list[int],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.tensor(centers_batch, dtype=torch.long, device=device)
    positives = torch.tensor(positives_batch, dtype=torch.long, device=device)
    return centers, positives


def generate_skipgram_batches(
    encoded_sentences: list[list[int]],
    window_size: int,
    batch_size: int,
    device: str | torch.device,
    rng: random.Random,
    keep_probabilities: list[float] | None = None,
    dynamic_window: bool = True,
):
    """Yield Skip-gram mini-batches without materializing all pairs in memory."""
    sentence_indices = list(range(len(encoded_sentences)))
    rng.shuffle(sentence_indices)

    centers_batch: list[int] = []
    positives_batch: list[int] = []

    for sentence_index in sentence_indices:
        sentence = subsample_sentence(encoded_sentences[sentence_index], keep_probabilities, rng)
        if len(sentence) < 2:
            continue
        for position, center in enumerate(sentence):
            current_window = rng.randint(1, window_size) if dynamic_window else window_size
            left = max(0, position - current_window)
            right = min(len(sentence), position + current_window + 1)
            context_words = sentence[left:position] + sentence[position + 1:right]

            for positive in context_words:
                centers_batch.append(center)
                positives_batch.append(positive)

                if len(centers_batch) == batch_size:
                    yield _collate_skipgram(centers_batch, positives_batch, device)
                    centers_batch, positives_batch = [], []

    if centers_batch:
        yield _collate_skipgram(centers_batch, positives_batch, device)


def build_model(model_type: str, vocab_size: int, embedding_dim: int, pad_idx: int) -> nn.Module:
    """Construct the requested PyTorch Word2Vec model."""
    if model_type == "cbow":
        return CBOWNegativeSamplingModel(vocab_size=vocab_size, embedding_dim=embedding_dim, pad_idx=pad_idx)
    if model_type == "skipgram":
        return SkipGramNegativeSamplingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    raise ValueError(f"Unsupported model type: {model_type}")


def train_word2vec_model(
    *,
    model_type: str,
    embedding_dim: int,
    window_size: int,
    negative_samples: int,
    encoded_sentences: list[list[int]],
    vocabulary: Vocabulary,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str | torch.device,
    seed: int,
    subsample_threshold: float = 1e-5,
    dynamic_window: bool = True,
) -> dict:
    """Train a Word2Vec model from scratch and return a serializable artifact."""
    if not encoded_sentences:
        raise ValueError("Encoded corpus is empty after vocabulary filtering.")

    pad_idx = vocabulary.size
    model = build_model(model_type=model_type, vocab_size=vocabulary.size, embedding_dim=embedding_dim, pad_idx=pad_idx)
    model = model.to(device)

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    sampler = NegativeSampler(vocabulary.counts, device=device)
    loss_history: list[float] = []
    examples_per_epoch = count_training_examples(encoded_sentences, window_size, model_type)
    total_tokens = sum(len(sentence) for sentence in encoded_sentences)
    keep_probabilities = None
    effective_subsample_threshold = 0.0
    subsampling_disabled_reason = "disabled_by_default"
    estimated_retained = float(total_tokens)

    if subsample_threshold > 0:
        candidate_keep_probabilities = build_subsampling_keep_probabilities(
            vocabulary.counts,
            total_tokens=total_tokens,
            threshold=subsample_threshold,
        )
        estimated_retained = expected_retained_tokens(encoded_sentences, candidate_keep_probabilities)
        avg_retained_per_sentence = estimated_retained / max(len(encoded_sentences), 1)

        # Small academic corpora collapse under the standard 1e-5 threshold.
        if total_tokens >= 500_000 and avg_retained_per_sentence >= 4.0:
            keep_probabilities = candidate_keep_probabilities
            effective_subsample_threshold = subsample_threshold
            subsampling_disabled_reason = ""
        else:
            subsampling_disabled_reason = "auto_disabled_for_small_corpus"
            estimated_retained = float(total_tokens)

    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        running_loss = 0.0
        batch_count = 0
        examples_seen = 0

        if model_type == "cbow":
            batch_iterator = generate_cbow_batches(
                encoded_sentences=encoded_sentences,
                window_size=window_size,
                batch_size=batch_size,
                pad_idx=pad_idx,
                device=device,
                rng=rng,
                keep_probabilities=keep_probabilities,
                dynamic_window=dynamic_window,
            )
            for contexts, mask, targets in batch_iterator:
                negatives = sampler.sample(targets.size(0), negative_samples, forbidden=targets)
                optimizer.zero_grad()
                loss = model(contexts, mask, targets, negatives)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batch_count += 1
                examples_seen += int(targets.size(0))
        else:
            batch_iterator = generate_skipgram_batches(
                encoded_sentences=encoded_sentences,
                window_size=window_size,
                batch_size=batch_size,
                device=device,
                rng=rng,
                keep_probabilities=keep_probabilities,
                dynamic_window=dynamic_window,
            )
            for centers, positives in batch_iterator:
                negatives = sampler.sample(centers.size(0), negative_samples, forbidden=positives)
                optimizer.zero_grad()
                loss = model(centers, positives, negatives)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batch_count += 1
                examples_seen += int(centers.size(0))

        average_loss = running_loss / max(batch_count, 1)
        loss_history.append(average_loss)
        examples_per_epoch = examples_seen

    input_weights = model.input_embeddings.weight.detach().cpu()
    if model_type == "cbow":
        input_weights = input_weights[: vocabulary.size]
    output_weights = model.output_embeddings.weight.detach().cpu()

    return {
        "model_type": model_type,
        "config": {
            "embedding_dim": embedding_dim,
            "window_size": window_size,
            "negative_samples": negative_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "subsample_threshold": effective_subsample_threshold,
            "dynamic_window": dynamic_window,
        },
        "training": {
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else math.nan,
            "examples_per_epoch": examples_per_epoch,
            "estimated_retained_tokens_per_epoch": round(estimated_retained, 2),
            "num_sentences": len(encoded_sentences),
            "vocab_size": vocabulary.size,
            "total_tokens": total_tokens,
            "subsampling_disabled_reason": subsampling_disabled_reason,
        },
        "vocab": {
            "itos": vocabulary.itos,
            "counts": vocabulary.counts,
        },
        "embeddings": {
            "input": input_weights,
            "output": output_weights,
        },
    }
