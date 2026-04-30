import json

from src.run_eval_from_logs import load_eval_examples


def test_load_eval_examples_filters_missing_expected(tmp_path):
    path = tmp_path / "chat_logs.json"
    path.write_text(
        json.dumps(
            [
                {"question": "Q1", "expected_substring": "A1", "ai_response": "A1"},
                {"question": "Q2", "expected_substring": "", "ai_response": "A2"},
                {"question": "", "expected_substring": "A3", "ai_response": "A3"},
            ]
        ),
        encoding="utf-8",
    )

    examples = load_eval_examples(str(path))

    assert len(examples) == 1
    assert examples[0].question == "Q1"
    assert examples[0].expected_substring == "A1"


def test_load_eval_examples_supports_custom_expected_field(tmp_path):
    path = tmp_path / "chat_logs.json"
    path.write_text(
        json.dumps(
            [
                {"question": "Q1", "benchmark": "Paris"},
                {"question": "Q2", "benchmark": ""},
            ]
        ),
        encoding="utf-8",
    )

    examples = load_eval_examples(str(path), expected_field="benchmark")

    assert len(examples) == 1
    assert examples[0].question == "Q1"
    assert examples[0].expected_substring == "Paris"
