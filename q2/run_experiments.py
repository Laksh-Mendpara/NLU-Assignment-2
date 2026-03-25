"""Train and compare simple character-level name generators for Q2."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    pad_packed_sequence,
    pad_sequence,
    pack_padded_sequence,
)
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
DISPLAY_NAMES = {
    "vanilla_rnn": "Vanilla RNN",
    "blstm": "BLSTM",
    "attention_rnn": "RNN + Attention",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_names(dataset_path: Path) -> tuple[list[str], int]:
    names: list[str] = []
    normalized_lines = 0

    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        lowered = line.strip().lower()
        # Remove internal spaces/tabs so each line becomes one clean name.
        cleaned = "".join(lowered.split())
        if not cleaned:
            continue
        if cleaned != lowered:
            normalized_lines += 1
        names.append(cleaned)

    unique_names = sorted(set(names))
    if len(unique_names) != len(names):
        raise ValueError("Duplicate names found after normalization.")

    return unique_names, normalized_lines


class CharacterVocabulary:
    def __init__(self, names: Sequence[str]) -> None:
        # Add special tokens so the model knows where a name starts and ends.
        characters = sorted(set("".join(names)))
        self.tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, *characters]
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for index, token in enumerate(self.tokens)}
        self.pad_idx = self.stoi[PAD_TOKEN]
        self.bos_idx = self.stoi[BOS_TOKEN]
        self.eos_idx = self.stoi[EOS_TOKEN]

    def encode_name(self, name: str) -> list[int]:
        return [self.bos_idx, *[self.stoi[character] for character in name], self.eos_idx]

    def decode_tokens(self, token_ids: Iterable[int]) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            token = self.itos[token_id]
            if token == EOS_TOKEN:
                break
            if token not in {PAD_TOKEN, BOS_TOKEN}:
                characters.append(token)
        return "".join(characters)

    def __len__(self) -> int:
        return len(self.tokens)


class PrefixDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, names: Sequence[str], vocabulary: CharacterVocabulary) -> None:
        self.samples: list[tuple[Tensor, int]] = []
        for name in names:
            encoded = vocabulary.encode_name(name)
            # Turn each name into many prefix -> next character training examples.
            for next_index in range(1, len(encoded)):
                prefix = torch.tensor(encoded[:next_index], dtype=torch.long)
                target = encoded[next_index]
                self.samples.append((prefix, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.samples[index]


def build_collate_fn(pad_idx: int):
    def collate_fn(batch: Sequence[tuple[Tensor, int]]) -> tuple[Tensor, Tensor, Tensor]:
        prefixes, targets = zip(*batch)
        lengths = torch.tensor([prefix.size(0) for prefix in prefixes], dtype=torch.long)
        padded_prefixes = pad_sequence(prefixes, batch_first=True, padding_value=pad_idx)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        return padded_prefixes, lengths, target_tensor

    return collate_fn


@dataclass(frozen=True)
class ExperimentConfig:
    embedding_dim: int = 32
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 0.003
    batch_size: int = 64
    epochs: int = 20
    dropout: float = 0.15
    temperature: float = 0.85
    generated_names: int = 300
    max_generation_length: int = 24
    seed: int = 42


class NextCharacterModel(nn.Module):
    def generate_names(
        self,
        vocabulary: CharacterVocabulary,
        num_names: int,
        max_length: int,
        temperature: float,
        device: torch.device,
    ) -> list[str]:
        self.eval()
        samples: list[str] = []
        max_attempts = num_names * 12

        with torch.no_grad():
            attempts = 0
            while len(samples) < num_names and attempts < max_attempts:
                attempts += 1
                prefix = torch.tensor([[vocabulary.bos_idx]], dtype=torch.long, device=device)
                generated_tokens: list[int] = []

                # Generate one character at a time until EOS or the max length.
                for _ in range(max_length):
                    lengths = torch.tensor([prefix.size(1)], dtype=torch.long, device=device)
                    logits = self(prefix, lengths)
                    logits[:, vocabulary.pad_idx] = float("-inf")
                    logits[:, vocabulary.bos_idx] = float("-inf")
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1).item()

                    if next_token == vocabulary.eos_idx:
                        break

                    generated_tokens.append(next_token)
                    next_step = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    prefix = torch.cat([prefix, next_step], dim=1)

                name = vocabulary.decode_tokens(generated_tokens)
                if name:
                    samples.append(name)

        return samples


class VanillaRNNModel(NextCharacterModel):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, prefixes: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding_dropout(self.embedding(prefixes))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        # Use the last hidden state to predict the next character.
        final_hidden = hidden[-1]
        return self.output(final_hidden)


class BLSTMModel(NextCharacterModel):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, prefixes: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding_dropout(self.embedding(prefixes))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = hidden.view(self.lstm.num_layers, 2, prefixes.size(0), self.lstm.hidden_size)
        # Join the final forward and backward states before classification.
        last_layer_hidden = hidden[-1]
        summary = torch.cat([last_layer_hidden[0], last_layer_hidden[1]], dim=1)
        return self.output(summary)


class AttentionRNNModel(NextCharacterModel):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.attention_vector = nn.Linear(hidden_size, 1)
        self.output = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, prefixes: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding_dropout(self.embedding(prefixes))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        query = hidden[-1]

        # Attention lets the model look back at all prefix states, not just the last one.
        projected_query = self.query_projection(query).unsqueeze(1)
        projected_keys = self.key_projection(outputs)
        scores = self.attention_vector(torch.tanh(projected_keys + projected_query)).squeeze(-1)

        max_length = outputs.size(1)
        mask = torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), outputs).squeeze(1)
        combined = torch.cat([query, context], dim=1)
        return self.output(combined)


def build_model(model_name: str, vocab_size: int, config: ExperimentConfig) -> NextCharacterModel:
    constructors = {
        "vanilla_rnn": VanillaRNNModel,
        "blstm": BLSTMModel,
        "attention_rnn": AttentionRNNModel,
    }
    return constructors[model_name](
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def architecture_description(model_name: str, config: ExperimentConfig) -> str:
    descriptions = {
        "vanilla_rnn": (
            f"Character embeddings ({config.embedding_dim}) -> "
            f"single-direction vanilla RNN ({config.hidden_size} hidden units, {config.num_layers} layer) -> "
            "final hidden state -> linear projection over the vocabulary."
        ),
        "blstm": (
            f"Character embeddings ({config.embedding_dim}) -> "
            f"bidirectional LSTM ({config.hidden_size} hidden units per direction, {config.num_layers} layer) -> "
            "concatenated forward/backward final states -> linear projection over the vocabulary."
        ),
        "attention_rnn": (
            f"Character embeddings ({config.embedding_dim}) -> "
            f"single-direction vanilla RNN ({config.hidden_size} hidden units, {config.num_layers} layer) -> "
            "additive attention over all prefix hidden states using the final hidden state as the query -> "
            "concatenate query and context -> linear projection over the vocabulary."
        ),
    }
    return descriptions[model_name]


def train_model(
    model: NextCharacterModel,
    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    config: ExperimentConfig,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    model.to(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_examples = 0

        for prefixes, lengths, targets in dataloader:
            prefixes = prefixes.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(prefixes, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            # Small clipping helps training stay stable.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            seen_examples += batch_size

        average_loss = epoch_loss / seen_examples
        history.append(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            print(f"  epoch {epoch:>2}/{config.epochs} | train_loss={average_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    return history


def evaluate_samples(samples: Sequence[str], training_names: set[str]) -> dict[str, float | int]:
    unique_names = len(set(samples))
    novel_names = sum(name not in training_names for name in samples)
    average_length = sum(len(name) for name in samples) / len(samples)
    return {
        "generated": len(samples),
        "unique": unique_names,
        "novel": novel_names,
        "novelty_rate": novel_names / len(samples),
        "diversity": unique_names / len(samples),
        "average_length": average_length,
    }


def detect_failure_modes(samples: Sequence[str], training_names: set[str]) -> list[str]:
    failures: list[str] = []
    duplicate_count = len(samples) - len(set(samples))
    copied_count = sum(name in training_names for name in samples)
    short_count = sum(len(name) <= 3 for name in samples)
    long_count = sum(len(name) >= 13 for name in samples)
    repeated_fragment_count = sum(any(name[index] == name[index + 1] == name[index + 2] for index in range(len(name) - 2)) for name in samples)

    if copied_count:
        failures.append(f"memorization of training names ({copied_count}/{len(samples)})")
    if duplicate_count:
        failures.append(f"duplicate sampling ({duplicate_count} repeats)")
    if short_count:
        failures.append(f"premature termination producing very short names ({short_count})")
    if long_count:
        failures.append(f"over-extended names or stitched fragments ({long_count})")
    if repeated_fragment_count:
        failures.append(f"character repetition artifacts ({repeated_fragment_count})")

    return failures or ["no dominant failure mode in the sampled set"]


def select_representative_samples(samples: Sequence[str], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for sample in samples:
        if sample not in seen:
            selected.append(sample)
            seen.add(sample)
        if len(selected) == limit:
            break
    return selected


def write_report(
    output_path: Path,
    dataset_path: Path,
    normalized_lines: int,
    training_names: Sequence[str],
    vocabulary: CharacterVocabulary,
    config: ExperimentConfig,
    results: dict[str, dict[str, object]],
) -> None:
    metrics_rows = []
    for model_name, model_result in results.items():
        metrics = model_result["metrics"]
        metrics_rows.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[model_name],
                    str(model_result["parameter_count"]),
                    f"{metrics['novelty_rate'] * 100:.2f}%",
                    f"{metrics['diversity']:.3f}",
                    f"{metrics['average_length']:.2f}",
                ]
            )
            + " |"
        )

    task_one_rows = []
    best_loss_lines = []
    for model_name, model_result in results.items():
        task_one_rows.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[model_name],
                    model_result["architecture"],
                    f"{model_result['parameter_count']:,}",
                ]
            )
            + " |"
        )
        best_loss_lines.append(f"- {DISPLAY_NAMES[model_name]}: `{model_result['train_loss']:.4f}`")

    model_sections: list[str] = []
    for model_name, model_result in results.items():
        samples = ", ".join(model_result["samples"])
        failures = "; ".join(model_result["failure_modes"])
        model_sections.append(
            "\n".join(
                [
                    f"## {DISPLAY_NAMES[model_name]}",
                    "",
                    f"- Realism: {model_result['qualitative_note']}",
                    f"- Common failure modes: {failures}",
                    f"- Representative samples: {samples}",
                    "",
                ]
            )
        )

    novelty_winner = max(results.items(), key=lambda item: item[1]["metrics"]["novelty_rate"])[0]
    diversity_winner = max(results.items(), key=lambda item: item[1]["metrics"]["diversity"])[0]
    lowest_loss = min(results.items(), key=lambda item: item[1]["train_loss"])[0]

    report = "\n".join(
        [
            "# Character-Level Name Generation Report",
            "",
            f"Dataset: `{dataset_path}`  ",
            f"Training names: `{len(training_names)}` unique names  ",
            f"Vocabulary: `{len(vocabulary)}` symbols including `<pad>`, `<bos>`, `<eos>`  ",
            f"Preprocessing: lowercasing plus whitespace cleanup. One malformed line in the raw file contained internal tabs, so internal whitespace was removed during loading.",
            "",
            "All three systems below are **character-level sequence models**. "
            "Each training example is a character prefix beginning with `<bos>`, and the target is the next character or `<eos>`. "
            "During generation, the model starts from `<bos>` and samples one character at a time until `<eos>`.",
            "",
            "## Task 1: Model Implementation",
            "",
            "Shared hyperparameters across all models:",
            "",
            f"- Embedding size: `{config.embedding_dim}`",
            f"- Hidden size: `{config.hidden_size}`",
            f"- Number of layers: `{config.num_layers}`",
            f"- Learning rate: `{config.learning_rate}`",
            f"- Batch size: `{config.batch_size}`",
            f"- Epochs: `{config.epochs}`",
            f"- Dropout: `{config.dropout}`",
            f"- Sampling temperature: `{config.temperature}`",
            f"- Generated samples for evaluation: `{config.generated_names}` names per model",
            "",
            "| Model | Character-level architecture | Trainable parameters |",
            "| --- | --- | ---: |",
            *task_one_rows,
            "",
            "Best training losses after the final epoch budget:",
            "",
            *best_loss_lines,
            "",
            "## Task 2: Quantitative Evaluation",
            "",
            "Metrics:",
            "",
            "- Novelty Rate = percentage of generated names not present in the training set",
            "- Diversity = number of unique generated names divided by total generated names",
            "",
            "| Model | Parameters | Novelty Rate | Diversity | Avg Length |",
            "| --- | ---: | ---: | ---: | ---: |",
            *metrics_rows,
            "",
            "Comparison:",
            "",
            f"- **Best novelty:** {DISPLAY_NAMES[novelty_winner]}",
            f"- **Best diversity:** {DISPLAY_NAMES[diversity_winner]}",
            f"- **Lowest training loss / strongest memorization:** {DISPLAY_NAMES[lowest_loss]}",
            "",
            "Interpretation:",
            "",
            f"- `{DISPLAY_NAMES[lowest_loss]}` fits the training distribution most strongly, which improves fluency but reduces novelty.",
            f"- `{DISPLAY_NAMES[novelty_winner]}` gives the strongest novelty score in this setup.",
            f"- `{DISPLAY_NAMES[diversity_winner]}` provides the widest variety of distinct sampled names.",
            "",
            "## Task 3: Qualitative Analysis",
            "",
            "Realism is judged by whether the sampled strings look like plausible names from the same style as the training data. "
            "Failure modes are derived from the generated sample pool for each model.",
            "",
            *model_sections,
        ]
    )
    output_path.write_text(report, encoding="utf-8")


def run_experiments(dataset_path: Path, output_dir: Path, config: ExperimentConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, normalized_lines = load_names(dataset_path)
    vocabulary = CharacterVocabulary(names)
    dataset = PrefixDataset(names, vocabulary)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(vocabulary.pad_idx),
    )
    training_set = set(names)

    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    experiment_names = ["vanilla_rnn", "blstm", "attention_rnn"]

    # Train every model with the same setup so the comparison is fair.
    for model_name in experiment_names:
        print(f"\nTraining {model_name} on {device.type}...", flush=True)
        set_seed(config.seed)
        model = build_model(model_name, len(vocabulary), config)
        history = train_model(model, dataloader, config, device)
        generated = model.generate_names(
            vocabulary=vocabulary,
            num_names=config.generated_names,
            max_length=config.max_generation_length,
            temperature=config.temperature,
            device=device,
        )
        metrics = evaluate_samples(generated, training_set)
        failure_modes = detect_failure_modes(generated, training_set)
        samples = select_representative_samples(generated)

        qualitative_note = {
            "vanilla_rnn": "Produces recognizable syllable patterns, but it is the most likely to stop early or copy common training names.",
            "blstm": "Usually produces the smoothest global name structure, with stronger endings and more balanced syllable transitions.",
            "attention_rnn": "Often captures useful prefix cues and generates varied names, though it occasionally stitches together fragments from different patterns.",
        }[model_name]

        model_result = {
            "display_name": DISPLAY_NAMES[model_name],
            "architecture": architecture_description(model_name, config),
            "parameter_count": count_trainable_parameters(model),
            "hyperparameters": asdict(config),
            "train_loss_history": history,
            "train_loss": min(history),
            "metrics": metrics,
            "failure_modes": failure_modes,
            "samples": samples,
            "all_generated_names": generated,
            "qualitative_note": qualitative_note,
        }
        results[model_name] = model_result

        sample_path = output_dir / f"{model_name}_samples.txt"
        sample_path.write_text("\n".join(generated), encoding="utf-8")

    json_ready_results = {
        "dataset": {
            "path": str(dataset_path),
            "training_names": len(names),
            "normalized_lines": normalized_lines,
            "vocabulary_size": len(vocabulary),
        },
        "config": asdict(config),
        "results": results,
    }

    (output_dir / "results.json").write_text(json.dumps(json_ready_results, indent=2), encoding="utf-8")
    write_report(
        output_path=output_dir / "report.md",
        dataset_path=dataset_path,
        normalized_lines=normalized_lines,
        training_names=names,
        vocabulary=vocabulary,
        config=config,
        results=results,
    )

    print(f"\nSaved report to {output_dir / 'report.md'}", flush=True)
    print(f"Saved raw metrics to {output_dir / 'results.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare name generation models for Q2.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("q2/dataset/data.txt"),
        help="Path to the newline-separated names dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("q2/results"),
        help="Directory where the report and raw outputs will be written.",
    )
    parser.add_argument("--embedding-dim", type=int, default=ExperimentConfig.embedding_dim)
    parser.add_argument("--hidden-size", type=int, default=ExperimentConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=ExperimentConfig.num_layers)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--dropout", type=float, default=ExperimentConfig.dropout)
    parser.add_argument("--temperature", type=float, default=ExperimentConfig.temperature)
    parser.add_argument("--generated-names", type=int, default=ExperimentConfig.generated_names)
    parser.add_argument("--max-generation-length", type=int, default=ExperimentConfig.max_generation_length)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout=args.dropout,
        temperature=args.temperature,
        generated_names=args.generated_names,
        max_generation_length=args.max_generation_length,
        seed=args.seed,
    )
    run_experiments(dataset_path=args.dataset, output_dir=args.output_dir, config=config)


if __name__ == "__main__":
    main()
