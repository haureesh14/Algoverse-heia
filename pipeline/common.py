# common.py: helpers shared by all the scripts below (settings, logging, OpenAI calls, resume cache)
import os, re, json, time, threading
import pandas as pd

# Load the settings saved in Step 5 and make sure the output folders exist
CFG = json.load(open("config.json"))
LANGS = CFG["langs"]                      # {"es": "Spanish", ...}  (English is the source)
os.makedirs("data/cache", exist_ok=True)
os.makedirs("outputs/gen", exist_ok=True)
os.makedirs("outputs/judged", exist_ok=True)
os.makedirs("outputs/judge_cache", exist_ok=True)

# Print a message with a UTC timestamp (these lines end up in run_all.log)
def log(msg):
    print(time.strftime("%H:%M:%S UTC", time.gmtime()), msg, flush=True)

# List every prompt file in data/ and read its language and translator ("variant") from the file name,
# e.g. prompts_hi_mt.csv -> ("hi", "mt"); prompts_en_orig.csv -> ("en", "orig")
def prompt_files():
    """data/prompts_{lang}_{variant}.csv -> [(lang, variant, path)]"""
    out = []
    for f in sorted(os.listdir("data")):
        m = re.fullmatch(r"prompts_([a-z]+)_([a-z0-9]+)\.csv", f)
        if m:
            out.append((m.group(1), m.group(2), os.path.join("data", f)))
    return out

# ---------- OpenAI helper (used by translate.py and judge.py) ----------
# Calls the OpenAI API directly over HTTPS with `requests` (no openai library), because the openai
# library installed on this machine crashes with "process() takes no keyword arguments"
import requests
API_URL = "https://api.openai.com/v1/chat/completions"

# Send one request to an OpenAI model and get back JSON
def chat_json(model, system, user, effort="minimal", max_tokens=1500, tries=8):
    """Returns (parsed_dict_or_None, raw_text). Retries rate limits / server errors with backoff."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None, "__ERROR__ OPENAI_API_KEY is not set (re-run Step 4)"
    # The request: system instructions + user message, a reasoning-effort level, and "reply in JSON"
    body = {"model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "reasoning_effort": effort, "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for i in range(tries):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=180)
        except requests.RequestException as e:          # network problem or timeout: worth retrying
            err = f"__ERROR__ {type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                # Pull the {...} part out of the reply and parse it; None if it isn't valid JSON
                raw = resp.json()["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", raw, flags=re.S)
                try:
                    return (json.loads(m.group(0)) if m else None), raw
                except Exception:
                    return None, raw
            err = f"__ERROR__ HTTP {resp.status_code}: {resp.text[:300]}"
            # Errors that retrying can't fix (bad request, wrong key, no access, unknown model, no credit): give up now
            if resp.status_code in (400, 401, 403, 404) or "insufficient_quota" in resp.text:
                return None, err
        # Temporary errors (rate limit, server error, network): wait 1, 2, 4, 8... seconds (max 60) and retry
        if i == tries - 1:
            return None, err
        time.sleep(min(60, 2 ** i))

# A file that remembers every finished API result (one JSON object per line), so a crash or a re-run
# continues where it stopped instead of paying for the same calls again
class JsonlCache:
    """Append-only cache so API steps can resume after a crash or restart."""
    # On start-up, read back everything already saved in the file
    def __init__(self, path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    # Skip results that were API errors, so they are tried again instead of reused
                    if str(d.get("issue", "")).startswith("api_error") or str(d.get("error", "")).startswith("__ERROR__"):
                        continue
                    self.data[d["key"]] = d
                except Exception:
                    pass                                # skip a half-written last line after a crash
    # "key in cache": has this item already been done?
    def __contains__(self, k):
        return k in self.data
    # Get a saved result
    def get(self, k):
        return self.data.get(k)
    # Save a new result to memory and append it to the file (the lock stops parallel threads clashing)
    def put(self, d):
        with self.lock:
            self.data[d["key"]] = d
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

# English refusal phrases, used to spot a translator that refused instead of translating
REFUSAL_EN = re.compile(r"(i can(no|')?t|i'?m sorry|i am sorry|i won'?t|i'?m unable|i am unable|"
                        r"cannot (help|assist|comply|translate|provide)|unable to (help|assist|translate))", re.I)
