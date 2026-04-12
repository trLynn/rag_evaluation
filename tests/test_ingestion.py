from src.ingestion import chunk_text


def test_chunk_text_produces_multiple_chunks() -> None:
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 3
    assert len(chunks[0]) == 500


def test_chunk_text_empty() -> None:
    assert chunk_text("   ") == []