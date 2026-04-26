from src.ingestion import chunk_documents_in_background


def test_chunk_documents_in_background(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world " * 20, encoding="utf-8")
    file_b.write_text("another file " * 20, encoding="utf-8")

    result = chunk_documents_in_background([str(file_b), str(file_a)], max_workers=2)

    assert [item.path.name for item in result] == ["a.txt", "b.txt"]
    assert all(item.chunks for item in result)