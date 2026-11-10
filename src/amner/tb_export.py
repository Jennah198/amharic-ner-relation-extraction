"""TensorBoard support.

* ``tb_report_to()`` – used by train_ner / train_mlm: logs to TensorBoard
  (``runs/<run>``) whenever the ``tensorboard`` package is installed.
* ``python -m amner.tb_export`` – converts the histories of runs that were
  trained *before* TensorBoard was enabled into TensorBoard event files, by
  parsing the metric dictionaries the Hugging Face Trainer printed to
  ``logs/ner_<run>.log`` and ``logs/mlm_dapt.log``. No retraining needed.

View:  tensorboard --logdir runs   ->  http://localhost:6006
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import shutil
from pathlib import Path

DICT_RE = re.compile(r"\{'[^{}]*\}")


def tb_report_to() -> list[str]:
    return ["tensorboard"] if importlib.util.find_spec("tensorboard") else []


def parse_log(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    out = []
    for m in DICT_RE.finditer(text):
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict) and "epoch" in d:
            out.append(d)
    return out


def export_run(writer_cls, name: str, records: list[dict], steps_per_epoch: int, out_root: Path) -> int:
    out = out_root / name
    if out.exists():
        shutil.rmtree(out)
    w = writer_cls(log_dir=str(out))
    n = 0
    for d in records:
        step = int(round(d["epoch"] * steps_per_epoch))
        for k, v in d.items():
            if k == "epoch" or not isinstance(v, (int, float)):
                continue
            if k.startswith("eval_"):
                tag = "eval/" + k[5:]
            elif k.startswith("train_"):
                tag = "train_summary/" + k[6:]
            else:
                tag = "train/" + k
            w.add_scalar(tag, v, step)
            n += 1
        w.add_scalar("train/epoch", d["epoch"], step)
    w.flush()
    w.close()
    return n


def main():
    from torch.utils.tensorboard import SummaryWriter

    out_root = Path("runs")
    for log in sorted(Path("logs").glob("ner_*.log")):
        name = log.stem[len("ner_"):]
        run = Path("reports/runs") / f"{name}.json"
        if not run.exists():
            continue
        spe = math.ceil(json.loads(run.read_text())["train_sentences"] / 16)  # batch size 16
        recs = parse_log(log)
        n = export_run(SummaryWriter, name, recs, spe, out_root)
        print(f"{name}: {len(recs)} log records, {n} scalars -> runs/{name}")
    mlm = Path("logs/mlm_dapt.log")
    if mlm.exists():
        recs = parse_log(mlm)
        rep = json.loads(Path("reports/mlm_dapt.json").read_text())
        spe = max(1, round(rep["steps"] / max(recs[-1]["epoch"], 1e-9))) if recs else 1
        n = export_run(SummaryWriter, "mlm_dapt", recs, spe, out_root)
        print(f"mlm_dapt: {len(recs)} log records, {n} scalars -> runs/mlm_dapt")


if __name__ == "__main__":
    main()
