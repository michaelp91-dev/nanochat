"""
Tests for training while the dataset is still downloading (base_train --wait-for-shards N).

python -m pytest tests/test_streaming_data.py -v
"""
import os
import tempfile
import threading
import time
_real_sleep = time.sleep

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

os.environ.setdefault("NANOCHAT_BASE_DIR", tempfile.mkdtemp(prefix="nanochat_stream_test_"))
from nanochat import dataset  # noqa: E402
from nanochat.dataloader import _document_batches  # noqa: E402


def write_shard(index, rows_per_group=4, groups=3, delay=0.0):
    time.sleep(delay)
    os.makedirs(dataset.DATA_DIR, exist_ok=True)
    path = os.path.join(dataset.DATA_DIR, dataset.index_to_filename(index))
    texts = [f"shard{index} doc{i}" for i in range(rows_per_group * groups)]
    pq.write_table(pa.table({"text": texts}), path + ".tmp", row_group_size=rows_per_group)
    os.rename(path + ".tmp", path)  # like download_single_file: atomic rename


@pytest.fixture(autouse=True)
def clean_dir():
    os.makedirs(dataset.DATA_DIR, exist_ok=True)
    for f in os.listdir(dataset.DATA_DIR):
        os.remove(os.path.join(dataset.DATA_DIR, f))
    yield


def test_waits_for_shards_in_order(monkeypatch):
    monkeypatch.setattr(dataset.time, "sleep", lambda s: _real_sleep(0.05))
    write_shard(dataset.MAX_SHARD)  # val first, like the downloader
    write_shard(0)
    def producer():
        for i in (1, 2):
            write_shard(i, delay=0.3)
        open(os.path.join(dataset.DATA_DIR, dataset.DONE_MARKER), "w").write("3\n")
    th = threading.Thread(target=producer)
    th.start()
    it = _document_batches("train", None, tokenizer_batch_size=100, wait_for_shards=3)
    seen = []
    for _ in range(9):  # 3 shards x 3 row groups
        texts, (pq_idx, rg_idx, epoch) = next(it)
        seen.append((pq_idx, rg_idx, epoch, texts[0].split()[0]))
    th.join()
    assert [s[3] for s in seen] == ["shard0"] * 3 + ["shard1"] * 3 + ["shard2"] * 3
    # next epoch starts over at shard 0
    texts, (pq_idx, rg_idx, epoch) = next(it)
    assert (pq_idx, epoch, texts[0].split()[0]) == (0, 2, "shard0")


def test_val_split_and_resume(monkeypatch):
    monkeypatch.setattr(dataset.time, "sleep", lambda s: _real_sleep(0.05))
    write_shard(dataset.MAX_SHARD)
    for i in range(3):
        write_shard(i)
    texts, _ = next(_document_batches("val", None, 100, wait_for_shards=3))
    assert texts[0].startswith(f"shard{dataset.MAX_SHARD}")
    # resume in the middle of shard 1 (row group 1 was the last one consumed)
    it = _document_batches("train", {"pq_idx": 1, "rg_idx": 1, "epoch": 1}, 100, wait_for_shards=3)
    texts, (pq_idx, rg_idx, epoch) = next(it)
    assert (pq_idx, rg_idx) == (1, 2) and texts[0].startswith("shard1")


def test_missing_shard_after_done_raises(monkeypatch):
    monkeypatch.setattr(dataset.time, "sleep", lambda s: _real_sleep(0.01))
    write_shard(0)
    open(os.path.join(dataset.DATA_DIR, dataset.DONE_MARKER), "w").write("2\n")
    it = _document_batches("train", None, 100, wait_for_shards=2)
    for _ in range(3):
        next(it)
    with pytest.raises(FileNotFoundError):
        next(it)


def test_default_mode_unchanged():
    write_shard(0)
    write_shard(dataset.MAX_SHARD)
    it = _document_batches("train", None, 100)
    texts, (pq_idx, rg_idx, epoch) = next(it)
    assert texts[0].startswith("shard0")
