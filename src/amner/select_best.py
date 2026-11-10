"""Pick the best NER run by *validation* F1 (never by test F1) and copy it to
``artifacts/ner/best`` – the model used by the API, the UI and the Space."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def main():
    runs = [json.loads(p.read_text()) for p in Path("reports/runs").glob("*.json")]
    best = max(runs, key=lambda r: r["best_val_f1"])
    src, dst = Path("artifacts/ner") / best["name"], Path("artifacts/ner/best")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "selected_run.json").write_text(json.dumps(best, indent=2))
    print(f"best by validation F1: {best['name']} (val {best['best_val_f1']}, test {best['test_f1']})")


if __name__ == "__main__":
    main()
