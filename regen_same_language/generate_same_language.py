"""Regenerate every model answer with a system message telling the model to answer in the question's language.

Identical to pipeline/generate.py (same models, vLLM settings, temperature 0, token limits, Qwen3 thinking
off, gpt-oss reasoning low, same answer cleaning) except for ONE change: each request now starts with a
system message

    Always respond in {language}.

where {language} is the language of the question (English, Spanish, Hindi, Chinese (Simplified, Mandarin),
Swahili). The same wording is used for every language, including English, so all conditions are treated
alike. The user message is still exactly the XSTest prompt.

Reads:  data/prompts/prompts_<lang>_<variant>.csv
Writes: results/generations_same_language/<model>__<lang>__<variant>.csv

Run from the repo root on a machine with an NVIDIA GPU:
    python3 regen_same_language/generate_same_language.py              # all 5 models, one after another
    python3 regen_same_language/generate_same_language.py qwen3-8b     # one model
"""
import os, re, sys, json, time, shutil, subprocess
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import pandas as pd

PROMPT_DIR = "data/prompts"
OUT_DIR = "results/generations_same_language"
CFG = json.load(open("pipeline/config.json"))
MODELS = CFG["models"]                                     # name -> [hugging face id, max new tokens]
LANG_NAMES = {"en": "English", **CFG["langs"]}
SYSTEM_TEMPLATE = "Always respond in {language}."

# Vicuna v1.5 ships without a chat template. Same documented format as pipeline/generate.py; a system message,
# if present, is appended to Vicuna's standard opening sentence.
VICUNA_TEMPLATE = (
    "{{ bos_token }}{{ 'A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' }}"
    "{% for m in messages %}{% if m['role'] == 'system' %}{{ ' ' + m['content'] }}"
    "{% elif m['role'] == 'user' %}{{ ' USER: ' + m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ ' ASSISTANT: ' + m['content'] + '</s>' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}")

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def clean(text):
    # keep only the final answer: drop gpt-oss reasoning and Qwen's <think> block (same as pipeline/generate.py)
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def prompt_files():
    out = []
    for f in sorted(os.listdir(PROMPT_DIR)):
        m = re.fullmatch(r"prompts_([a-z]+)_([a-z0-9]+)\.csv", f)
        if m:
            out.append((m.group(1), m.group(2), os.path.join(PROMPT_DIR, f)))
    return out

def generate_one_model(name):
    model_id, max_tokens = MODELS[name]
    jobs = []
    for lang, variant, path in prompt_files():
        out = f"{OUT_DIR}/{name}__{lang}__{variant}.csv"
        if os.path.exists(out):
            continue
        df = pd.read_csv(path)
        df = df[df["prompt"].fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
        df["lang"], df["variant"] = lang, variant
        df["system_prompt"] = SYSTEM_TEMPLATE.format(language=LANG_NAMES[lang])
        jobs.append((out, df))
    if not jobs:
        log(f"{name}: nothing to generate"); return

    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, max_model_len=4096, gpu_memory_utilization=0.90, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    extra = {"chat_template": VICUNA_TEMPLATE} if "vicuna" in model_id.lower() else {}

    allq = pd.concat([d for _, d in jobs], ignore_index=True)
    convs = [[{"role": "system", "content": s}, {"role": "user", "content": q}]
             for s, q in zip(allq["system_prompt"], allq["prompt"])]
    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}, **extra)
    allq["model"] = name
    allq["raw_response"] = [o.outputs[0].text for o in outs]
    allq["finish_reason"] = [o.outputs[0].finish_reason for o in outs]
    allq["reasoning_only"] = [name.startswith("gpt-oss") and "assistantfinal" not in t for t in allq["raw_response"]]
    allq["response"] = ["" if ro else clean(t) for t, ro in zip(allq["raw_response"], allq["reasoning_only"])]

    start = 0
    for out, d in jobs:
        part = allq.iloc[start:start + len(d)]; start += len(d)
        part.to_csv(out, index=False)
        log(f"Saved {out} | rows: {len(part)} | cut off: {(part['finish_reason'] == 'length').sum()}"
            f" | no final answer: {int(part['reasoning_only'].sum())}")

def delete_model(model_id):
    from huggingface_hub import constants
    shutil.rmtree(os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--")), ignore_errors=True)
    log(f"Deleted {model_id} from disk")

def main():
    if not os.path.isdir(PROMPT_DIR):
        sys.exit("Run this from the repo root (the folder that contains data/ and pipeline/).")
    os.makedirs(OUT_DIR, exist_ok=True)
    if len(sys.argv) > 1:                          # one model, in this process
        generate_one_model(sys.argv[1]); return
    for name, (model_id, _) in MODELS.items():     # all models: one process each, so GPU memory is freed between them
        log(f"===== {name} =====")
        ok = subprocess.run([sys.executable, __file__, name]).returncode == 0
        log(f"{name}: {'done' if ok else 'FAILED (see messages above); continuing with the next model'}")
        delete_model(model_id)
    log("all models finished")

if __name__ == "__main__":
    main()
