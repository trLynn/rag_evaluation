class _FlakyCollection:
    def __init__(self):
        self.calls = []

    def upsert(self, ids, documents, metadatas):
        self.calls.append(len(ids))
        if len(ids) > 5:
            raise RuntimeError("timed out in upsert")


class _RetryThenPassCollection:
    def __init__(self):
        self.calls = 0

    def upsert(self, ids, documents, metadatas):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("ReadTimeout")


def test_chunk_documents_in_background(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world " * 20, encoding="utf-8")
    file_b.write_text("another file " * 20, encoding="utf-8")

    result = chunk_documents_in_background([str(file_b), str(file_a)], max_workers=2)

    assert [item.path.name for item in result] == ["a.txt", "b.txt"]
    assert all(item.chunks for item in result)


def test_upsert_batch_with_backoff_splits_timeout_batch():
    ids = [str(i) for i in range(12)]
    docs = [f"doc-{i}" for i in range(12)]
    metas = [{"chunk": i} for i in range(12)]
    collection = _FlakyCollection()

    _upsert_batch_with_backoff(
        collection=collection,
        batch_ids=ids,
        batch_chunks=docs,
        batch_metadatas=metas,
        min_batch_size=5,
        base_retry_sleep_seconds=0,
    )

    # Initial call fails at size 12 and then should recursively split to <= 5.
    assert collection.calls[0] == 12
    assert any(call_size <= 5 for call_size in collection.calls)


def test_upsert_batch_with_backoff_retries_before_split():
    collection = _RetryThenPassCollection()
    _upsert_batch_with_backoff(
        collection=collection,
        batch_ids=["1", "2"],
        batch_chunks=["a", "b"],
        batch_metadatas=[{"chunk": 0}, {"chunk": 1}],
        max_timeout_retries=3,
        base_retry_sleep_seconds=0,
    )
    assert collection.calls == 3
