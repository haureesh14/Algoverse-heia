"""Run one model over every prompt file in data/ that it hasn't answered yet (loads the model once).
Usage: python generate.py NAME HF_MODEL_ID MAX_TOKENS"""
import os
# Run vLLM in this same process (more reliable inside JupyterHub)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import sys
from common import *

# Vicuna v1.5 ships without a chat template; this is its documented prompt format.
# Result: "<s>A chat between ... questions. USER: <prompt> ASSISTANT:"
VICUNA_TEMPLATE = (
    "{{ bos_token }}{{ 'A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' }}"
    "{% for m in messages %}{% if m['role'] == 'user' %}{{ ' USER: ' + m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ ' ASSISTANT: ' + m['content'] + '</s>' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}")

# Keep only the final answer: drop gpt-oss's reasoning (before "assistantfinal") and Qwen's <think> block
def clean(text):
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def main():
    # Read the arguments: short name, Hugging Face model id, max answer length
    name, model_id, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
    # Collect every prompt file this model hasn't answered yet (skip finished ones and empty prompts)
    jobs = []
    for lang, variant, path in prompt_files():
        out = f"outputs/gen/{name}__{lang}__{variant}.csv"
        if os.path.exists(out):
            continue
        df = pd.read_csv(path)
        df = df[df["prompt"].fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
        df["lang"], df["variant"] = lang, variant
        jobs.append((out, df))
    if not jobs:
        log(f"{name}: nothing to generate"); return

    # Load the model once. temperature=0 = always pick the most likely word (repeatable answers)
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, max_model_len=4096, gpu_memory_utilization=0.90, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    # Vicuna needs the template above; other models have their own built in
    extra = {"chat_template": VICUNA_TEMPLATE} if "vicuna" in model_id.lower() else {}

    # Put all languages into one big batch (much faster), each prompt as a single user message.
    # enable_thinking=False turns off Qwen's thinking mode; reasoning_effort="low" keeps gpt-oss short
    allq = pd.concat([d for _, d in jobs], ignore_index=True)
    outs = llm.chat([[{"role": "user", "content": q}] for q in allq["prompt"]], params,
                    chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}, **extra)
    # Store the raw output, why it stopped ("length" = hit the token limit), and the cleaned final answer.
    # reasoning_only = gpt-oss ran out of tokens while still reasoning, so there is no answer
    allq["model"] = name
    allq["raw_response"] = [o.outputs[0].text for o in outs]
    allq["finish_reason"] = [o.outputs[0].finish_reason for o in outs]
    allq["reasoning_only"] = [name.startswith("gpt-oss") and "assistantfinal" not in t for t in allq["raw_response"]]
    allq["response"] = ["" if ro else clean(t) for t, ro in zip(allq["raw_response"], allq["reasoning_only"])]

    # Split the big batch back into one file per language/translator: outputs/gen/{model}__{lang}__{variant}.csv
    start = 0
    for out, d in jobs:
        part = allq.iloc[start:start + len(d)]
        start += len(d)
        part.to_csv(out, index=False)
        log(f"Saved {out} | rows: {len(part)} | cut off: {(part['finish_reason'] == 'length').sum()}"
            f" | no final answer: {int(part['reasoning_only'].sum())}")

if __name__ == "__main__":
    main()
