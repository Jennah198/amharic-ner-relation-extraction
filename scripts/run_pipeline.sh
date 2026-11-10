#!/usr/bin/env bash
# Reproduce the whole project end-to-end. Each stage is also runnable alone.
# CPU-only machine: ~2-3 h in total. With a GPU most training steps take minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
BASE=${BASE:-rasyosef/roberta-medium-amharic}

echo "== 1. Data collection"
python -m amner.collect.hf_benchmarks                       # MasakhaNER amh + Hub news text
python -m amner.collect.bbc_amharic --max-articles 600       # scraped BBC Amharic
# python -m amner.collect.telegram_channels --channels <names>  # optional, needs .env

echo "== 2. Normalization & cleaning"
python -m amner.build_corpus
python -m amner.build_lexicon

echo "== 3. Tokenization"
python -m amner.tokenization --base-model "$BASE"

echo "== 4. Model adaptation"
python -m amner.train_mlm --base-model "$BASE" --out artifacts/mlm_dapt --name dapt --max-steps 800
python -m amner.train_ner --model "$BASE"             --name base_full
python -m amner.train_ner --model artifacts/mlm_dapt  --name dapt_full
python -m amner.train_ner --model artifacts/mlm_dapt  --name dapt_lora --mode lora
python -m amner.train_ner --model artifacts/mlm_dapt  --name dapt_full_morph --morph-augment
python -m amner.train_ner --model "$BASE"             --name base_full_morph --morph-augment

echo "== 5. Evaluation"
for m in base_full base_full_morph dapt_full dapt_lora dapt_full_morph; do
  python -m amner.evaluate --model artifacts/ner/$m --name $m
done
python -m amner.select_best
python -m amner.evaluate_re --model artifacts/ner/best
python -m amner.optimize --model artifacts/ner/best
python -m amner.summarize

echo "== 6. Serve:  uvicorn api.main:app --port 8000  &&  AMNER_API_URL=http://127.0.0.1:8000 python app/gradio_app.py"
