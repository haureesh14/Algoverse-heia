"""Whole pipeline, safe to re-run: every step skips work that is already saved.
translate (OpenAI) -> NLLB check -> generate with each model (judging that model in the background) -> judge leftovers.
Backs up to Hugging Face after every step."""
import os, sys, shutil, subprocess
from huggingface_hub import HfApi, constants
from common import *

api = HfApi()

# Upload data/, outputs/, the scripts, settings and log to the Hugging Face backup dataset.
# A failed upload is logged but never stops the pipeline
def backup(msg):
    try:
        for folder in ("data", "outputs"):
            api.upload_folder(repo_id=CFG["hf_repo"], repo_type="dataset", folder_path=folder, path_in_repo=folder,
                              commit_message=msg, ignore_patterns=["*.ipynb_checkpoints*"])
        for f in ("common.py", "translate.py", "nllb_translate.py", "generate.py", "judge.py", "run_all.py",
                  "config.json", "run_all.log"):
            if os.path.exists(f):
                api.upload_file(repo_id=CFG["hf_repo"], repo_type="dataset", path_or_fileobj=f,
                                path_in_repo=f, commit_message=msg)
        log(f"Backed up: {msg}")
    except Exception as e:
        log(f"BACKUP FAILED ({msg}): {e}")

# Delete a downloaded model from disk once we're done with it (the disk is shared with other teams)
def delete_model(model_id):
    shutil.rmtree(os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--")), ignore_errors=True)
    log(f"Deleted {model_id} from disk")

# Run one of our scripts as a separate process, wait for it, and report success/failure
def run(args):
    log("Running: " + " ".join(args))
    ok = subprocess.run([sys.executable] + args).returncode == 0
    if not ok:
        log("FAILED: " + " ".join(args))
    return ok

def main():
    log("===== pipeline start =====")
    # 1) Translate all languages with OpenAI, then back up
    if run(["translate.py"]):
        backup("translations")
    # 2) NLLB second-translator sample on the GPU, back up, then free the disk space
    if CFG["run_nllb"]:
        if run(["nllb_translate.py"]):
            backup("nllb translations")
        delete_model(CFG["nllb_model"])

    # 3) For each model: generate answers, back up, delete the model, then start judging that model's
    #    answers in the background while the next model is generating
    judges = []
    for name, (model_id, max_tokens) in CFG["models"].items():
        if run(["generate.py", name, model_id, str(max_tokens)]):
            backup(f"generations: {name}")
        delete_model(model_id)
        log(f"Judging {name} in the background")
        judges.append(subprocess.Popen([sys.executable, "judge.py", name]))
    # 4) Wait for all background judges, judge anything they missed, final backup
    for p in judges:
        p.wait()
    run(["judge.py"])          # anything left over (e.g. a background judge that crashed)
    backup("judgments")
    log("===== pipeline DONE =====")

if __name__ == "__main__":
    main()
