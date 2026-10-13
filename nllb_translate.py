"""Second-translator check: translate a fixed sample of XSTest with NLLB-200 (free, open, never refuses).
Writes data/prompts_{lang}_nllb.csv for the same prompt ids in every language."""
from common import *

# NLLB's codes for our languages (language + script)
CODES = {"es": "spa_Latn", "hi": "hin_Deva", "zh": "zho_Hans", "sw": "swh_Latn"}

def main():
    # Pick the same random sample every time (random_state=0): 100 safe + 50 unsafe prompts
    en = pd.read_csv("data/prompts_en_orig.csv")
    n_safe, n_unsafe = CFG["nllb_sample"]
    sample = pd.concat([en[en["label"] == "safe"].sample(n_safe, random_state=0),
                        en[en["label"] == "unsafe"].sample(n_unsafe, random_state=0)]).sort_values("id")
    # Only do languages that don't have an NLLB file yet
    todo = [l for l in LANGS if l in CODES and not os.path.exists(f"data/prompts_{l}_nllb.csv")]
    if not todo:
        log("NLLB: nothing to do"); return
    # Load the NLLB model onto the GPU (half precision to save memory)
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(CFG["nllb_model"], src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG["nllb_model"], torch_dtype=torch.float16).cuda().eval()
    texts = sample["prompt"].tolist()
    for lang in todo:
        # Translate 16 prompts at a time; forced_bos_token_id tells NLLB which language to write in
        outs = []
        for i in range(0, len(texts), 16):
            batch = tok(texts[i:i + 16], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids(CODES[lang]),
                                     num_beams=4, max_new_tokens=256)
            outs += tok.batch_decode(gen, skip_special_tokens=True)
        # Save in the same format as the GPT translations, marked variant "nllb"
        df = sample[["id", "label", "type", "prompt"]].rename(columns={"prompt": "prompt_en"})
        df["prompt"], df["lang"], df["variant"] = outs, lang, "nllb"
        df.to_csv(f"data/prompts_{lang}_nllb.csv", index=False)
        log(f"NLLB {lang}: saved {len(df)} rows")

if __name__ == "__main__":
    main()
