# Multilingual Over-Refusal (Algoverse)

Do safety-tuned LLMs refuse harmless requests more often when they are asked in another language?
We translated [XSTest](https://github.com/paul-rottger/xstest) (250 safe + 200 unsafe prompts) from English into
Spanish, Hindi, Chinese and Swahili, ran five open models on every version, and labelled each answer as
compliance / refusal with an LLM judge.

| | |
|---|---|
| **Languages** | English (source), Spanish, Hindi, Chinese (Simplified), Swahili |
| **Models tested** | Qwen3-8B, Qwen3-14B, Llama-3.1-8B-Instruct, Vicuna-7B v1.5, gpt-oss-20b (via vLLM, greedy decoding) |
| **Translator** | GPT-5 mini (with back-translation + meaning check); NLLB-200 and Qwen2.5-14B as second translators |
| **Judge** | GPT-5 mini, XSTest classes: full compliance / full refusal / partial refusal / other |

## Folders

```
notebooks/
  multilingual_pipeline.ipynb     main notebook (Oct 8 run): setup, run, quality checks, analysis
  xstest_english_eval.ipynb       earlier English-only pilot (Sep 17)
  xstest_spanish_eval.ipynb       earlier Spanish pilot (Sep 17)
pipeline/                         scripts the notebook writes and runs
  common.py  translate.py  nllb_translate.py  generate.py  judge.py  run_all.py  config.json
data/
  prompts/                        prompts_<lang>_<translator>.csv
                                  translator: orig = English original, mt = GPT-5 mini,
                                  nllb = NLLB-200 (150-prompt sample), qwen = Qwen2.5-14B (Spanish, Sep 17)
  translation_cache/              raw translator outputs (lets translate.py resume)
results/
  generations/                    <model>__<lang>__<translator>.csv  raw model answers
  judged/                         same files + judge label (judgment column)  <- use these for analysis
  judge_cache/                    raw judge outputs (lets judge.py resume)
  analysis/                       tables from notebook Step 13/14 (refusal rates, paired tests, translator check)
  hand_labels/                    hand_label_v2.csv: 120 sampled answers for human labelling (NOT yet labelled)
                                  hand_label_v2_key.csv: the judge's labels for those rows (keep hidden from labellers)
logs/
  run_all.log                     full pipeline log
archive/
  mlor_results.zip                snapshot of everything above as backed up to Hugging Face
```

## Known data issues (read before using the numbers)

- **Translator refusals:** GPT-5 mini refused to translate 83 harmful prompts (es 26, hi 28, zh 11, sw 18) and
  wrote a refusal instead. Notebook Step 13 detects and excludes them (`results/analysis/translator_refusals.csv`).
- **Vicuna truncated refusals:** 212 Vicuna answers are just `Lo sient` (a cut-off "Lo siento") and 3 are
  `No es aprop`; the judge labelled them `other`. They should be counted as refusals.
- **Judge misses some refusals:** answers that decline and then offer help resources are sometimes labelled
  compliance, most often in Chinese. Treat unsafe-prompt refusal rates as lower bounds until the judge is validated.
- **Hand labels pending:** the judge has not been validated against human labels yet.
