from src.evaluation import EvalExample, evaluate_retrieval


def test_evaluate_retrieval_hit_rate(monkeypatch) -> None:
    def fake_retrieve_context(question: str, **_: object):
        if "capital" in question:
            return ["Paris is the capital of France"]
        return ["Unknown"]

    monkeypatch.setattr("src.evaluation.retrieve_context", fake_retrieve_context)

    result = evaluate_retrieval(
        [
            EvalExample(question="What is the capital of France?", expected_substring="Paris"),
            EvalExample(question="What is 2+2?", expected_substring="4"),
        ]
    )

    assert result.total == 2
    assert result.hits == 1
    assert result.hit_rate == 0.5