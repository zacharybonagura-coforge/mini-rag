import json
from pathlib import Path

from adapters import build_embedder, build_generator, build_store
from config import load_settings
from generate import generate
from ingest import ingest

QUESTIONS = [
    "How much can I spend on food each day?",
    "Can I book first-class airfare?",
    "My hotel costs $250. What do I need?",
    "Do I need a receipt for a $20 taxi?",
    "Can I claim a limousine upgrade?",
    "Does the company reimburse gym memberships?",
]


def main() -> None:
    settings = load_settings()
    embedder = build_embedder(settings)
    generator = build_generator(settings)
    store = build_store(settings)

    if not store.load_chunks():
        ingest(Path(settings.policy_path), embedder, store)

    results = []
    for question in QUESTIONS:
        response = generate(
            question,
            embedder,
            generator,
            store,
            k=settings.retrieve_k,
        )
        record = {
            "question": question,
            **response.model_dump(),
        }
        results.append(record)
        print(question)
        print(json.dumps(record, indent=2))
        print()

    output_path = Path(settings.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
