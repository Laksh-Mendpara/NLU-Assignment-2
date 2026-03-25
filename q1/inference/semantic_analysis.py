from __future__ import annotations

"""Run simple semantic checks on the trained Q1 embeddings.

The comments here explain why each scoring and filtering step exists.
"""

import argparse
import glob
import json
import os
import sys

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from word2vec.artifacts import load_artifact, nearest_neighbors, solve_analogy


ANALYSIS_EXCLUDED_TOKENS = set(ENGLISH_STOP_WORDS) | {
    "iit",
    "iitj",
    "jodhpur",
    "indian",
    "institute",
    "technology",
    "dr",
    "mr",
    "mrs",
    "ms",
    "clicking",
    "contact",
    "please",
    "page",
    "link",
    "webpages",
    "booklet",
    "batch",
    "student_id",
    "gen",
    "obc",
    "ews",
    "pwd",
    "ra",
    "ta",
    "th",
    "july",
    "november",
    "pdf",
    "dcs",
    "cete",
    "jpd",
    "nh",
    "bb",
    "es",
    "mumbai",
    "poland",
    "texas",
    "subrata",
    "cet",
    "drc",
    "jam",
    "ma",
    "metre",
    "ope",
    "pp",
    "post",
    "major",
    "written",
    "additional",
    "admitted",
    "approval",
    "barc",
    "bhabha",
    "minutes",
    "calendar",
    "certificate",
    "dates",
    "december",
    "deficient",
    "awarded",
    "appeal",
    "followed",
    "bonafide",
    "gandhinagar",
    "granted",
    "issued",
    "january",
    "june",
    "letter",
    "marks",
    "mid",
    "months",
    "moe",
    "orientation",
    "paid",
    "permission",
    "prescribed",
    "registered",
    "ready",
    "seek",
    "specified",
    "stay",
    "touch",
    "transcript",
    "vacation",
    "withdrawal",
    "warning",
    "waitlisted",
    "websites",
    "movement",
    "nation",
    "national",
    "therapeutic",
    "biss",
    "wise",
    "allowing",
    "necessary",
    "bangalore",
    "sanjay",
    "guwahati",
    "card",
    "wf",
    "jee",
    "shall",
    "sop",
    "cat",
    "colloquium",
    "elect",
    "enrol",
    "mp",
    "october",
    "prescribe",
    "provisionally",
    "qualify",
    "register",
    "representative",
    "secretary",
    "specify",
    "technology_jodhpur",
    "verify",
    "verification",
    "vizag",
    "tifr",
    "ualbany",
    "wsom",
    "edtech",
    "preparedness",
    "reproducible",
    "translatable",
    "flagship",
    "guarantee",
    "cancellation",
    "shortlist",
    "sponsore",
    "rajesh",
    "dindorkar",
    "undertake",
    "expand",
    "spread",
    "direction",
    "thematic",
    "talent",
    "excellent",
    "wish",
    "absence",
    "tailor",
    "organize",
    "empower",
    "person",
    "reason",
    "travel",
    "able",
    "admit",
    "chairman",
    "committee",
    "duration",
    "earn",
    "fee",
    "france",
    "futuristic",
    "institution",
    "initiative",
    "leaf",
    "month",
    "pay",
    "profile",
    "repute",
    "track",
    "visit",
}

ANALYSIS_ALLOWED_COMPOUNDS = {
    "comprehensive_exam",
    "faculty_member",
    "grade_credit",
    "compulsory_course",
    "postgraduate_program",
    "research_board",
    "program_elective",
    "phd_program",
    "mtech_phd",
    "undergraduate_program",
    "office_of_students",
    "student_life",
    "student_gymkhana",
    "research_development",
}

ANALYSIS_ALWAYS_KEEP = {
    "research",
    "student",
    "hostel",
    "phd",
    "exam",
    "course",
    "faculty",
    "department",
    "btech",
    "mtech",
    "ug",
    "pg",
    "undergraduate",
    "postgraduate",
    "fellowship",
    "assistantship",
    "thesis",
    "publication",
    "project",
    "lab",
    "campus",
    "room",
    "dining",
    "allocation",
    "facility",
    "recreational",
}

ANALYSIS_MIN_COUNT = 6

ANALOGY_EXPERIMENTS = [
    {
        "a": "undergraduate",
        "b": "btech",
        "c": "postgraduate",
        "expected": ["mtech", "mba", "msc", "mtech_phd"],
        "relation": "program level -> example degree",
    },
    {
        "a": "btech",
        "b": "undergraduate",
        "c": "mtech",
        "expected": ["postgraduate"],
        "relation": "degree -> program level",
    },
    {
        "a": "student",
        "b": "gymkhana",
        "c": "faculty",
        "expected": ["member", "staff"],
        "relation": "community -> representative body / membership",
    },
    {
        "a": "mtech",
        "b": "postgraduate",
        "c": "btech",
        "expected": ["ug", "undergraduate"],
        "relation": "degree -> program level",
    },
    {
        "a": "semester",
        "b": "course",
        "c": "phd",
        "expected": ["comprehensive_exam"],
        "relation": "academic stage -> key academic requirement",
    },
    {
        "a": "faculty",
        "b": "department",
        "c": "room",
        "expected": ["hostel"],
        "relation": "entity -> parent unit / container",
    },
]


def parse_args() -> argparse.Namespace:
    return argparse.Namespace(
        models_dir="output/models",
        model_paths=None,
        topn=5,
        analogy_topn=5,
        embedding_mode="combined",
    )


def select_default_model_paths(models_dir: str) -> list[str]:
    """Pick the strongest CBOW/Skip-gram pair from experiment metrics, then fall back to stable defaults."""
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

                def ranking_key(record: dict) -> tuple:
                    # Lower validation loss is better, so it comes first in the ranking.
                    best_validation_loss = record.get("best_validation_loss")
                    if best_validation_loss is None or best_validation_loss != best_validation_loss:
                        best_validation_loss = float("inf")
                    retained_ratio = record.get("retained_token_ratio_per_epoch")
                    if retained_ratio is None:
                        total_tokens = record.get("total_tokens") or 0
                        retained_tokens = record.get("estimated_retained_tokens_per_epoch") or 0
                        # Formula:
                        # retained_ratio = retained_tokens_per_epoch / total_tokens
                        retained_ratio = (retained_tokens / total_tokens) if total_tokens else 0.0
                    return (
                        best_validation_loss,
                        record.get("final_loss", float("inf")),
                        0 if record.get("window_size") == 2 else 1,
                        -float(retained_ratio),
                        0 if record.get("negative_samples") == 10 else 1,
                        record.get("embedding_dim", 0),
                    )

                best = min(candidates, key=ranking_key)
                artifact_path = best.get("artifact_path")
                if artifact_path and os.path.exists(artifact_path):
                    selected.append(artifact_path)
            if selected:
                return selected
        except Exception:
            pass

    preferred_by_architecture = {
        "cbow": [
            os.path.join(models_dir, "cbow_dim300_win2_neg10.pt"),
            os.path.join(models_dir, "cbow_dim200_win2_neg10.pt"),
            os.path.join(models_dir, "cbow_dim300_win5_neg10.pt"),
            os.path.join(models_dir, "cbow_dim200_win5_neg10.pt"),
            os.path.join(models_dir, "cbow_dim200_win2_neg5.pt"),
            os.path.join(models_dir, "cbow_dim300_win2_neg5.pt"),
            os.path.join(models_dir, "cbow_dim200_win5_neg5.pt"),
            os.path.join(models_dir, "cbow_dim300_win5_neg5.pt"),
            os.path.join(models_dir, "cbow_dim100_win2_neg5.pt"),
            os.path.join(models_dir, "cbow_dim100_win5_neg5.pt"),
        ],
        "skipgram": [
            os.path.join(models_dir, "skipgram_dim200_win2_neg10.pt"),
            os.path.join(models_dir, "skipgram_dim300_win2_neg10.pt"),
            os.path.join(models_dir, "skipgram_dim200_win5_neg10.pt"),
            os.path.join(models_dir, "skipgram_dim300_win5_neg10.pt"),
            os.path.join(models_dir, "skipgram_dim200_win2_neg5.pt"),
            os.path.join(models_dir, "skipgram_dim300_win2_neg5.pt"),
            os.path.join(models_dir, "skipgram_dim200_win5_neg5.pt"),
            os.path.join(models_dir, "skipgram_dim300_win5_neg5.pt"),
            os.path.join(models_dir, "skipgram_dim100_win2_neg5.pt"),
            os.path.join(models_dir, "skipgram_dim100_win5_neg5.pt"),
        ],
    }
    preferred_selected: list[str] = []
    for architecture in ("cbow", "skipgram"):
        for path in preferred_by_architecture[architecture]:
            if os.path.exists(path):
                preferred_selected.append(path)
                break
    if preferred_selected:
        return preferred_selected

    paths = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    selected: list[str] = []
    for architecture in ("cbow", "skipgram"):
        for path in paths:
            if os.path.basename(path).startswith(f"{architecture}_"):
                selected.append(path)
                break
    return selected


def build_analysis_exclusions(artifact: dict) -> set[str]:
    # This removes obvious junk so the semantic neighbors are easier to read.
    excluded = set(ANALYSIS_EXCLUDED_TOKENS)
    tokens = artifact["vocab"]["itos"]
    counts = artifact["vocab"].get("counts", [0] * len(tokens))

    # Remove obvious junk tokens so the nearest-neighbor lists are easier to interpret.
    for token, count in zip(tokens, counts):
        if token in ANALYSIS_ALWAYS_KEEP:
            continue
        if count < ANALYSIS_MIN_COUNT:
            excluded.add(token)
            continue
        if len(token) <= 2 and token not in {"ug", "pg", "phd", "ai"}:
            excluded.add(token)
            continue
        if len(token) > 24:
            excluded.add(token)
            continue
        if any(ch.isdigit() for ch in token):
            excluded.add(token)
            continue
        if any(ch in token for ch in "&/+"):
            excluded.add(token)
            continue
        if "_" in token and token not in ANALYSIS_ALLOWED_COMPOUNDS:
            excluded.add(token)

    return excluded


def run_semantic_analysis(model_path: str, topn: int, analogy_topn: int, embedding_mode: str) -> dict:
    # This function prints a readable report and also returns JSON-ready data.
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

    analysis_exclusions = build_analysis_exclusions(artifact)

    target_words = ["research", "student", "hostel", "phd", "exam"]
    analysis = {"model_path": model_path, "model_label": label, "neighbors": {}, "analogies": {}}

    print(f"\n--- 1. Top {topn} Nearest Neighbors ---")
    # These words are the ones I used to inspect whether the embeddings learned useful campus semantics.
    for word in target_words:
        neighbors = nearest_neighbors(
            artifact,
            word,
            topn=topn,
            mode=embedding_mode,
            exclude_tokens=analysis_exclusions,
        )
        if not neighbors:
            print(f"\nWord '{word}' not in vocabulary.")
            analysis["neighbors"][word] = []
            continue

        print(f"\nNeighbors for '{word}':")
        for neighbor, similarity in neighbors:
            print(f"  - {neighbor} (cos: {similarity:.4f})")
        analysis["neighbors"][word] = neighbors

    print("\n--- 2. Analogy Experiments ---")
    # Analogies are a quick sanity check for relation-like structure in the embedding space.
    for analogy in ANALOGY_EXPERIMENTS:
        a = analogy["a"]
        b = analogy["b"]
        c = analogy["c"]
        expected = analogy["expected"]
        predictions = solve_analogy(
            artifact,
            a,
            b,
            c,
            topn=analogy_topn,
            mode=embedding_mode,
            exclude_tokens=analysis_exclusions,
        )
        key = f"{a}:{b}::{c}:?"
        if not predictions:
            print(f"\nSkipping analogy '{a}' : '{b}' :: '{c}' : ? (missing vocabulary items)")
            analysis["analogies"][key] = {"expected": expected, "relation": analogy["relation"], "predictions": []}
            continue

        print(f"\nAnalogy: '{a}' is to '{b}' as '{c}' is to ?")
        print(f"  expected relation: {analogy['relation']}")
        print(f"  expected answers: {', '.join(expected)}")
        for word, similarity in predictions:
            print(f"  -> {word} (cos: {similarity:.4f})")
        matched = next((word for word, _ in predictions if word in expected), None)
        if matched:
            print(f"  note: matched expected answer '{matched}'")
        else:
            print("  note: no exact expected answer in top predictions")
        analysis["analogies"][key] = {
            "expected": expected,
            "relation": analogy["relation"],
            "predictions": predictions,
        }

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
    # Save the console results as JSON so they can be included in the report later.
    report_path = os.path.join(report_dir, "semantic_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSemantic analysis report saved to {report_path}")


if __name__ == "__main__":
    main()
