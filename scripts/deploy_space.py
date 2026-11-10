"""Publish the model to the Hugging Face Hub and deploy the Gradio app as a Space.

Prerequisite:  huggingface-cli login   (a token with *write* access)

    python scripts/deploy_space.py --model-repo amharic-ner-roberta-morph \
                                   --space amharic-ner-relation-extraction

The Space contains app.py (the same Gradio UI), the amner package and a
requirements file; at start-up it downloads the model from the model repo.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Amharic NER & Relation Extraction
emoji: 🇪🇹
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "{gradio_version}"
app_file: app.py
pinned: false
license: mit
---

Morphology-aware Amharic named-entity recognition (PER, ORG, LOC, DATE) and political
relation extraction. Source code: {github}
"""

SPACE_REQS = """torch==2.5.1
transformers==4.46.3
tokenizers>=0.20,<0.21
spacy==3.8.2
requests
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="artifacts/ner/best")
    ap.add_argument("--model-repo", default="amharic-ner-roberta-morph")
    ap.add_argument("--space", default="amharic-ner-relation-extraction")
    ap.add_argument("--github", default="https://github.com/Jennah198/amharic-ner-relation-extraction")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    import gradio
    api = HfApi()
    user = api.whoami()["name"]
    model_id, space_id = f"{user}/{a.model_repo}", f"{user}/{a.space}"

    api.create_repo(model_id, repo_type="model", exist_ok=True, private=a.private)
    model_card = ROOT / "space" / "MODEL_CARD.md"
    if model_card.exists():
        shutil.copy(model_card, ROOT / a.model_dir / "README.md")
    api.upload_folder(repo_id=model_id, folder_path=str(ROOT / a.model_dir), repo_type="model",
                      commit_message="Upload fine-tuned Amharic NER model")
    print(f"model  -> https://huggingface.co/{model_id}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "src" / "amner", tmp / "amner",
                        ignore=shutil.ignore_patterns("__pycache__", "collect"))
        app = (ROOT / "app" / "gradio_app.py").read_text()
        header = ("import os\nos.environ.setdefault('AMNER_MODEL_DIR', '%s')\n" % model_id)
        (tmp / "app.py").write_text(header + app.replace('demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))', "demo.launch()"))
        (tmp / "requirements.txt").write_text(SPACE_REQS)
        (tmp / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__, github=a.github))
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=a.private)
        api.upload_folder(repo_id=space_id, folder_path=str(tmp), repo_type="space",
                          commit_message="Deploy Amharic NER & RE demo")
    print(f"space  -> https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
