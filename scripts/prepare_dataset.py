"""Combine labeled FEN files and create reproducible train/test splits."""

from collections import Counter, defaultdict
from pathlib import Path
import random

LABELS = {
    "Nothing",
    "Check White",
    "Check Black",
    "Checkmate White",
    "Checkmate Black",
}


def main():
    project_root = Path(__file__).resolve().parent.parent
    source = project_root / "dataset"
    output = project_root / "data"
    output.mkdir(exist_ok=True)

    files = sorted(source.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in {source}")

    positions = {}
    duplicates = 0

    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            for number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue

                parts = raw.split()
                if len(parts) < 7:
                    raise ValueError(
                        f"{path}:{number}: expected FEN followed by label"
                    )

                fen = " ".join(parts[:6])
                label = " ".join(parts[6:])

                if label not in LABELS:
                    raise ValueError(
                        f"{path}:{number}: unknown label {label!r}"
                    )

                # The encoder ignores the last two FEN fields (move counters).
                # Deduplicate using exactly the fields the model sees.
                key = " ".join(parts[:4])

                if key in positions:
                    if positions[key][1] != label:
                        raise ValueError(
                            f"{path}:{number}: conflicting labels for {key}"
                        )
                    duplicates += 1
                    continue

                positions[key] = (fen, label)

    groups = defaultdict(list)
    for fen, label in positions.values():
        groups[label].append(f"{fen} {label}")

    rng = random.Random(42)
    train, test = [], []

    for label in sorted(LABELS):
        rows = groups[label]
        if len(rows) < 2:
            raise ValueError(
                f"Need at least two unique positions for {label!r}; "
                f"found {len(rows)}"
            )

        rng.shuffle(rows)
        test_count = max(1, int(len(rows) * 0.2))

        test.extend(rows[:test_count])
        train.extend(rows[test_count:])

    rng.shuffle(train)
    rng.shuffle(test)

    for name, rows in (("train.txt", train), ("test.txt", test)):
        (output / name).write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

        counts = Counter(" ".join(row.split()[6:]) for row in rows)
        print(f"{name}: {len(rows)} positions")
        for label in sorted(LABELS):
            print(f"  {label}: {counts[label]}")

    print(f"Removed {duplicates} duplicate positions")


if __name__ == "__main__":
    main()