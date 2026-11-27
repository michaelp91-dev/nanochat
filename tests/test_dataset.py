"""
Env overrides and private-dataset auth for nanochat/dataset.py.

ClimbMix stays the default when the variables are unset. python -m pytest tests/test_dataset.py -v
"""
import os

import pytest

from nanochat import dataset
from nanochat.common import get_base_dir

TINYFORGE_URL = "https://huggingface.co/datasets/michaelp91-dev/tinyforge-lever1-mix/resolve/main"
TOKEN = "hf_super_secret_token_do_not_print"


def _clear_overrides(monkeypatch):
    for name in ("NANOCHAT_DATA_BASE_URL", "NANOCHAT_MAX_SHARD", "NANOCHAT_DATA_DIR", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


class _FakeResponse:
    def __init__(self, payload=b"parquet-bytes"):
        self.payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        yield self.payload


def test_climbmix_defaults_when_unset(monkeypatch):
    _clear_overrides(monkeypatch)
    assert dataset.get_base_url() == dataset.DEFAULT_BASE_URL
    assert dataset.get_max_shard() == 6542
    assert dataset.DEFAULT_BASE_URL == "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
    assert dataset.BASE_URL == dataset.DEFAULT_BASE_URL
    assert dataset.MAX_SHARD == 6542
    assert dataset.get_data_dir() == os.path.join(get_base_dir(), "base_data_climbmix")
    assert dataset.DATA_DIR == dataset.get_data_dir()


def test_blank_env_falls_back_to_climbmix(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DATA_BASE_URL", "  ")
    monkeypatch.setenv("NANOCHAT_MAX_SHARD", "")
    monkeypatch.setenv("NANOCHAT_DATA_DIR", "   ")
    monkeypatch.setenv("HF_TOKEN", "  ")
    assert dataset.get_base_url() == dataset.DEFAULT_BASE_URL
    assert dataset.get_max_shard() == dataset.DEFAULT_MAX_SHARD
    assert dataset.get_data_dir() == os.path.join(get_base_dir(), "base_data_climbmix")
    assert dataset._auth_headers() == {}
    monkeypatch.setenv("NANOCHAT_DATA_DIR", "~/tinyforge-data")
    assert dataset.get_data_dir() == os.path.expanduser("~/tinyforge-data")


def test_env_overrides_url_shard_and_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOCHAT_DATA_BASE_URL", TINYFORGE_URL + "/")
    monkeypatch.setenv("NANOCHAT_MAX_SHARD", "131")
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(tmp_path))
    assert dataset.get_base_url() == TINYFORGE_URL
    assert dataset.BASE_URL == TINYFORGE_URL
    assert dataset.MAX_SHARD == 131
    assert dataset.DATA_DIR == str(tmp_path)

    train = dataset.streaming_parquet_paths("train", 131)
    assert len(train) == 131
    assert train[0] == os.path.join(str(tmp_path), "shard_00000.parquet")
    assert train[-1] == os.path.join(str(tmp_path), "shard_00130.parquet")
    assert dataset.streaming_parquet_paths("val", 131) == [os.path.join(str(tmp_path), "shard_00131.parquet")]
    # an explicit directory argument still wins over the env var
    assert dataset.streaming_parquet_paths("val", 1, data_dir="/other") == ["/other/shard_00131.parquet"]

    num_train, ids = dataset.iter_shard_indices(8)
    assert (num_train, ids) == (8, [131, 0, 1, 2, 3, 4, 5, 6, 7])
    num_train, ids = dataset.iter_shard_indices(-1)
    assert num_train == 131 and ids == [131] + list(range(131))

    open(os.path.join(tmp_path, "shard_00000.parquet"), "wb").close()
    open(os.path.join(tmp_path, "ignore.tmp"), "wb").close()
    assert dataset.list_parquet_files() == [os.path.join(str(tmp_path), "shard_00000.parquet")]


def test_download_sends_bearer_token_and_does_not_print_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("NANOCHAT_DATA_BASE_URL", TINYFORGE_URL)
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(tmp_path))
    captured = {}

    def fake_get(url, stream=None, timeout=None, headers=None):
        captured["url"] = url
        captured["stream"] = stream
        captured["timeout"] = timeout
        captured["headers"] = dict(headers or {})
        return _FakeResponse(b"shard-bytes")

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    assert dataset.download_single_file(0) is True
    assert captured == {
        "url": f"{TINYFORGE_URL}/shard_00000.parquet",
        "stream": True,
        "timeout": 30,
        "headers": {"Authorization": f"Bearer {TOKEN}"},
    }
    assert open(os.path.join(tmp_path, "shard_00000.parquet"), "rb").read() == b"shard-bytes"
    output = capsys.readouterr().out
    assert TOKEN not in output
    assert "Authorization" not in output
    assert "Bearer" not in output


def test_download_omits_auth_header_without_token(monkeypatch, tmp_path, capsys):
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(tmp_path))
    captured = {}

    def fake_get(url, stream=None, timeout=None, headers=None):
        captured["headers"] = dict(headers or {})
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    assert dataset.download_single_file(2) is True
    assert captured["headers"] == {}
    assert captured["url"] == f"{dataset.DEFAULT_BASE_URL}/shard_00002.parquet"
    assert "Authorization" not in capsys.readouterr().out


def test_download_failure_redacts_token(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dataset.time, "sleep", lambda seconds: None)

    def fake_get(url, stream=None, timeout=None, headers=None):
        raise dataset.requests.RequestException(f"401 for {url} token={TOKEN}")

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    assert dataset.download_single_file(4) is False
    output = capsys.readouterr().out
    assert TOKEN not in output
    assert "***" in output
    assert not os.path.exists(os.path.join(tmp_path, "shard_00004.parquet"))
    assert not os.path.exists(os.path.join(tmp_path, "shard_00004.parquet.tmp"))


def test_download_skips_existing_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NANOCHAT_DATA_DIR", str(tmp_path))
    path = os.path.join(tmp_path, "shard_00001.parquet")
    open(path, "wb").write(b"already")

    def fake_get(*args, **kwargs):
        raise AssertionError("should not download a shard that is already on disk")

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    assert dataset.download_single_file(1) is True
    assert open(path, "rb").read() == b"already"
    assert "already exists" in capsys.readouterr().out
