"""
Offline tests for nanochat/hf_hub.py (checkpoint push/pull). The Hugging Face Hub is
replaced by a local fake, so no token or network is needed.

python -m pytest tests/test_hf_hub.py -v
"""
import os
import shutil
import tempfile

import huggingface_hub
import pytest

os.environ["NANOCHAT_BASE_DIR"] = tempfile.mkdtemp(prefix="nanochat_hf_test_")
from nanochat import hf_hub  # noqa: E402
from nanochat.common import get_base_dir  # noqa: E402


class _NotFound(Exception):
    pass


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    root = tmp_path / "hub"

    class FakeApi:
        def __init__(self, *a, **k): pass
        def create_repo(self, repo, private=False, exist_ok=False, **k):
            assert private
            (root / repo).mkdir(parents=True, exist_ok=True)
        def list_repo_files(self, repo, **k):
            d = root / repo
            if not d.exists():
                raise _NotFound()
            return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
        def create_commit(self, repo, ops, commit_message="", **k):
            for op in ops:
                dst = root / repo / op.path_in_repo
                if isinstance(op, huggingface_hub.CommitOperationDelete):
                    dst.unlink()
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(op.path_or_fileobj, dst)
        def super_squash_history(self, **k): pass

    def fake_download(repo, filename, local_dir=None, **k):
        dst = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(root / repo / filename, dst)
        return dst

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub.utils, "RepositoryNotFoundError", _NotFound)
    monkeypatch.setattr(hf_hub.subprocess, "Popen", lambda *a, **k: None)  # run push inline instead
    monkeypatch.setattr(hf_hub.time, "sleep", lambda s: None)
    return FakeApi()


def fake_checkpoint(d, step, content=b"x"):
    os.makedirs(d, exist_ok=True)
    for name in hf_hub.checkpoint_files(d, step)[:2] + [f"optim_{step:06d}_rank0.pt"]:
        open(os.path.join(d, name), "wb").write(content + name.encode())


def test_push_prune_pull(fake_hub, tmp_path):
    ckpt = str(tmp_path / "ckpt")
    tok = os.path.join(get_base_dir(), "tokenizer")
    os.makedirs(tok, exist_ok=True)
    open(os.path.join(tok, "tokenizer.pkl"), "wb").write(b"tok")
    repo = "me/nanochat-test"
    for step in (5, 10):
        fake_checkpoint(ckpt, step)
        hf_hub.stage_and_push(ckpt, step, repo)
        hf_hub.prune_checkpoints(ckpt, 1)  # prune before the upload ran: staged hardlinks must survive
    # both staged, the first push uploads only the newest
    assert hf_hub.push(repo, os.path.join(ckpt, ".hf_push", "step_000005")) == 0
    files = fake_hub.list_repo_files(repo)
    assert files == ["ckpt/meta_000010.json", "ckpt/model_000010.pt", "ckpt/optim_000010_rank0.pt", "tokenizer/tokenizer.pkl"], files
    assert hf_hub.local_steps(ckpt) == [10]
    # next checkpoint replaces the old one on the hub
    fake_checkpoint(ckpt, 15, b"y")
    hf_hub.stage_and_push(ckpt, 15, repo)
    assert hf_hub.wait(ckpt, repo) == 0
    assert [f for f in fake_hub.list_repo_files(repo) if f.startswith("ckpt/")] == \
        ["ckpt/meta_000015.json", "ckpt/model_000015.pt", "ckpt/optim_000015_rank0.pt"]
    # pull into a fresh dir with a different local tokenizer: the hub's wins
    open(os.path.join(tok, "tokenizer.pkl"), "wb").write(b"other")
    fresh = str(tmp_path / "fresh")
    assert hf_hub.pull(fresh, repo) == 15
    assert hf_hub.local_steps(fresh) == [15]
    assert open(os.path.join(fresh, "model_000015.pt"), "rb").read() == b"ymodel_000015.pt"
    assert open(os.path.join(tok, "tokenizer.pkl"), "rb").read() == b"tok"
    assert hf_hub.pull(str(tmp_path / "x"), "me/missing") == -1
