"""
Push base_train checkpoints to a (private) Hugging Face model repo in the background,
and pull the latest one back to resume, e.g. after a Colab disconnect.

base_train --hf-repo user/repo [--hf-push-every STEPS] stages each saved checkpoint
(model_XXXXXX.pt, meta_XXXXXX.json, optim_XXXXXX_rank*.pt, plus the tokenizer) into
<checkpoint_dir>/.hf_push/step_XXXXXX with hardlinks and starts

    python -m nanochat.hf_hub push --repo user/repo --staging <that dir>

in the background, so training never waits. The files go to ckpt/ (and tokenizer/) of the
repo, which is created private if missing; older ckpt/ files are deleted in the same commit
and the history is squashed, so the repo only holds the latest checkpoint. Pushes are
serialized with a lock; if a newer checkpoint was staged meanwhile, only the newest is pushed.

    python -m nanochat.hf_hub wait --checkpoint-dir DIR --repo user/repo   # wait for pending pushes
    python -m nanochat.hf_hub pull --checkpoint-dir DIR --repo user/repo   # restore latest ckpt + tokenizer
    python -m nanochat.hf_hub latest --checkpoint-dir DIR                  # print latest local step (-1 if none)

The token comes from the HF_TOKEN environment variable (or a cached `hf auth login`) and is
never printed. Set NANOCHAT_HF_DRY_RUN=1 to print what would be uploaded instead.
"""

import argparse
import fcntl
import glob
import os
import re
import shutil
import subprocess
import sys
import time

from nanochat.common import get_base_dir

CKPT_RE = re.compile(r"^(model_(\d{6})\.pt|meta_(\d{6})\.json|optim_(\d{6})_rank\d+\.pt)$")


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[hf_hub]", *a, flush=True)


def checkpoint_files(checkpoint_dir, step):
    tag = f"{step:06d}"
    names = [f"model_{tag}.pt", f"meta_{tag}.json"]
    names += sorted(os.path.basename(p) for p in glob.glob(os.path.join(checkpoint_dir, f"optim_{tag}_rank*.pt")))
    return names


def local_steps(checkpoint_dir):
    # steps with both a model and a meta file (meta is written after the model)
    steps = []
    for p in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
        m = re.match(r"model_(\d{6})\.pt$", os.path.basename(p))
        if m and os.path.exists(os.path.join(checkpoint_dir, f"meta_{m.group(1)}.json")):
            steps.append(int(m.group(1)))
    return sorted(steps)


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)  # keeps the data alive even if the checkpoint is pruned mid-upload
    except OSError:
        shutil.copy2(src, dst)


# -----------------------------------------------------------------------------
# called from base_train (master process) right after save_checkpoint

def stage_and_push(checkpoint_dir, step, repo):
    staging = os.path.join(checkpoint_dir, ".hf_push", f"step_{step:06d}")
    os.makedirs(staging, exist_ok=True)
    names = checkpoint_files(checkpoint_dir, step)
    meta = names.pop(1)
    for name in names:
        _link_or_copy(os.path.join(checkpoint_dir, name), os.path.join(staging, name))
    tok_dir = os.path.join(get_base_dir(), "tokenizer")
    if os.path.isdir(tok_dir):
        os.makedirs(os.path.join(staging, "tokenizer"), exist_ok=True)
        for name in os.listdir(tok_dir):
            if os.path.isfile(os.path.join(tok_dir, name)):
                _link_or_copy(os.path.join(tok_dir, name), os.path.join(staging, "tokenizer", name))
    _link_or_copy(os.path.join(checkpoint_dir, meta), os.path.join(staging, meta))  # meta last = complete
    logf = open(os.path.join(checkpoint_dir, "hf_push.log"), "a")
    subprocess.Popen([sys.executable, "-m", "nanochat.hf_hub", "push", "--repo", repo, "--staging", staging],
                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Pushing checkpoint at step {step} to hf://{repo} in the background (log: {checkpoint_dir}/hf_push.log)")


def prune_checkpoints(checkpoint_dir, keep):
    """ keep only the newest `keep` local checkpoints (staged pushes hold their own hardlinks) """
    for step in local_steps(checkpoint_dir)[:-keep]:
        for name in checkpoint_files(checkpoint_dir, step):
            try:
                os.remove(os.path.join(checkpoint_dir, name))
            except FileNotFoundError:
                pass


# -----------------------------------------------------------------------------
# background uploader

def _staged(push_dir):
    out = []
    for d in glob.glob(os.path.join(push_dir, "step_*")):
        m = re.match(r"step_(\d+)$", os.path.basename(d))
        if m and glob.glob(os.path.join(d, "meta_*.json")):
            out.append((int(m.group(1)), d))
    return sorted(out)


def _upload(repo, folder, step, dry_run):
    files = [(os.path.join(folder, f), f"ckpt/{f}") for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f))]
    tok = os.path.join(folder, "tokenizer")
    if os.path.isdir(tok):
        files += [(os.path.join(tok, f), f"tokenizer/{f}") for f in sorted(os.listdir(tok))]
    if dry_run:
        log(f"[dry-run] would create private repo {repo} (if missing), upload {[r for _, r in files]}, "
            f"delete older ckpt/ files and squash history")
        return
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi()  # token from HF_TOKEN / cached login
    api.create_repo(repo, private=True, exist_ok=True)
    new = {r for _, r in files}
    old = [f for f in api.list_repo_files(repo) if f.startswith("ckpt/") and f not in new]
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=p) for p, r in files]
    ops += [CommitOperationDelete(path_in_repo=f) for f in old]
    api.create_commit(repo, ops, commit_message=f"checkpoint step {step}")
    try:
        api.super_squash_history(repo_id=repo, commit_message=f"latest checkpoint: step {step}")
    except Exception as e:
        log(f"squash failed (not fatal): {type(e).__name__}: {e}")


def push(repo, staging, dry_run=False):
    push_dir = os.path.dirname(os.path.abspath(staging))
    os.makedirs(push_dir, exist_ok=True)
    with open(os.path.join(push_dir, "lock"), "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        staged = _staged(push_dir)
        if not staged:
            log("nothing staged (already pushed)")
            return 0
        step, folder = staged[-1]
        for _, d in staged[:-1]:
            log(f"skipping older staged checkpoint {os.path.basename(d)}, a newer one is ready")
            shutil.rmtree(d, ignore_errors=True)
        for attempt in range(1, 4):
            try:
                t0 = time.time()
                _upload(repo, folder, step, dry_run)
                log(f"pushed step {step} to hf://{repo} in {time.time() - t0:.0f}s")
                shutil.rmtree(folder, ignore_errors=True)
                return 0
            except Exception as e:
                log(f"push of step {step} failed (attempt {attempt}/3): {type(e).__name__}: {e}")
                time.sleep(30 * attempt)
        log(f"giving up on step {step} for now; it stays staged and is retried by the next push or `wait`")
        return 1


def wait(checkpoint_dir, repo, dry_run=False):
    push_dir = os.path.join(checkpoint_dir, ".hf_push")
    if not os.path.isdir(push_dir):
        return 0
    while True:
        with open(os.path.join(push_dir, "lock"), "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)  # waits for a running upload
            staged = _staged(push_dir)
        if not staged:
            log("no pending pushes")
            return 0
        if push(repo, staged[-1][1], dry_run) != 0:
            return 1


def pull(checkpoint_dir, repo, dry_run=False):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        files = HfApi().list_repo_files(repo)
    except RepositoryNotFoundError:
        log(f"repo {repo} not found (or no access): fresh start")
        return -1
    steps = sorted(int(m.group(1)) for f in files if (m := re.match(r"ckpt/meta_(\d{6})\.json$", f)))
    if not steps:
        log(f"no checkpoint in {repo}: fresh start")
        return -1
    step = steps[-1]
    have = local_steps(checkpoint_dir)
    tag = f"{step:06d}"
    wanted = [f for f in files if re.match(rf"ckpt/(model_{tag}\.pt|optim_{tag}_rank\d+\.pt)$", f)]
    wanted.append(f"ckpt/meta_{tag}.json")  # last
    tok = [f for f in files if f.startswith("tokenizer/")]
    if dry_run:
        log(f"[dry-run] would download {wanted + tok}")
        return step
    # the tokenizer must match the checkpoint, so always take the hub's
    tok_dir = os.path.join(get_base_dir(), "tokenizer")
    staging = os.path.join(checkpoint_dir, ".hf_pull")
    os.makedirs(tok_dir, exist_ok=True)
    for f in tok:
        os.replace(hf_hub_download(repo, f, local_dir=staging), os.path.join(tok_dir, os.path.basename(f)))
    if have and have[-1] >= step:
        log(f"{checkpoint_dir} already has step {have[-1]} (hub has {step})")
        return have[-1]
    os.makedirs(checkpoint_dir, exist_ok=True)
    for f in wanted:
        log(f"downloading {f}")
        os.replace(hf_hub_download(repo, f, local_dir=staging), os.path.join(checkpoint_dir, os.path.basename(f)))
    shutil.rmtree(staging, ignore_errors=True)
    log(f"restored step {step} into {checkpoint_dir}; resume with --resume-from-step={step}")
    return step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["push", "wait", "pull", "latest"])
    ap.add_argument("--repo")
    ap.add_argument("--staging")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--dry-run", action="store_true", default=os.environ.get("NANOCHAT_HF_DRY_RUN") == "1")
    args = ap.parse_args()
    if args.command == "push":
        sys.exit(push(args.repo, args.staging, args.dry_run))
    if args.command == "wait":
        sys.exit(wait(args.checkpoint_dir, args.repo, args.dry_run))
    if args.command == "pull":
        pull(args.checkpoint_dir, args.repo, args.dry_run)
        return
    steps = local_steps(args.checkpoint_dir)
    print(steps[-1] if steps else -1)


if __name__ == "__main__":
    main()
