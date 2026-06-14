from pathlib import Path
import json

import anyio
from langchain_core.runnables import RunnableLambda

from app.eval.metrics import (
    FaithfulnessJudge,
    FaithfulnessJudgment,
    data_grounding,
    expected_data_points,
    guardrail_correctness,
    source_attribution,
)
from app.eval.test_set import load_test_set
from app.rag.guardrails import EATING_DISORDER_RESPONSE, MEDICAL_RESPONSE
from app.rag.models import GuardrailCategory, RetrievalSource


def test_evaluation_set_has_planned_case_counts() -> None:
    test_set = load_test_set(Path("evaluation-test-set.json"))

    assert len(test_set.rag_cases) == 5
    assert len(test_set.analysis_cases) == 5
    assert len(test_set.agent_cases) == 3
    assert len(test_set.guardrail_cases) == 2
    assert len({case.case_id for case in test_set.all_cases}) == 15


def test_load_test_set_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "test-set.json"
    path.write_text(
        json.dumps(
            {
                "rag_cases": [
                    {
                        "case_id": "duplicate",
                        "question": "Question one?",
                        "expected_topic": "topic",
                        "expected_source_doc": "source.md",
                    }
                ],
                "analysis_cases": [],
                "agent_cases": [],
                "guardrail_cases": [
                    {
                        "case_id": "duplicate",
                        "question": "Question two?",
                        "expected_category": "medical",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_test_set(path)
    except ValueError as exc:
        assert "duplicates: duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate case IDs must be rejected")


def test_load_test_set_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "test-set.json"
    path.write_text('{"rag_cases": [}', encoding="utf-8")

    try:
        load_test_set(path)
    except ValueError as exc:
        assert "not valid JSON" in str(exc)
        assert str(path) in str(exc)
    else:
        raise AssertionError("Invalid JSON must be rejected")


def test_load_test_set_rejects_allowed_guardrail_expectation(tmp_path: Path) -> None:
    path = tmp_path / "test-set.json"
    path.write_text(
        json.dumps(
            {
                "rag_cases": [],
                "analysis_cases": [],
                "agent_cases": [],
                "guardrail_cases": [
                    {
                        "case_id": "not_blocked",
                        "question": "How should I warm up?",
                        "expected_category": "allowed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_test_set(path)
    except ValueError as exc:
        assert "must expect a blocked category" in str(exc)
    else:
        raise AssertionError("Allowed guardrail expectations must be rejected")


def test_source_attribution_requires_existing_document_and_known_chunk(tmp_path: Path) -> None:
    (tmp_path / "01-bench-press.md").write_text("# Bench", encoding="utf-8")
    source = RetrievalSource(
        source_file="01-bench-press.md",
        section_title="Safety",
        chunk_id="01-bench-press.md::0001",
    )

    passing = source_attribution(
        [source],
        knowledge_base_dir=tmp_path,
        known_chunk_ids={"01-bench-press.md::0001"},
    )
    failing = source_attribution(
        [source],
        knowledge_base_dir=tmp_path,
        known_chunk_ids={"01-bench-press.md::0000"},
    )

    assert passing.passed is True
    assert failing.passed is False


def test_data_grounding_matches_summary_numbers_and_human_readable_dates() -> None:
    summary = {
        "strength": {"percent_change": 17.86, "end_date": "2026-03-17"},
        "mixed_units_normalized": False,
    }

    result = data_grounding(
        "Estimated strength rose 17.86% through March 17, 2026.",
        summary,
    )

    assert result.passed is True
    assert {"17.86", "2026-03-17"}.issubset(result.details["matched_values"])


def test_expected_data_points_resolve_paths_and_report_mismatches() -> None:
    summary = {"exercise_trends": {"Bench Press": {"likely_deload_dates": ["2026-01-27"]}}}

    passing = expected_data_points(
        summary,
        [("exercise_trends.Bench Press.likely_deload_dates.0", "2026-01-27")],
    )
    failing = expected_data_points(summary, [("exercise_trends.Bench Press.session_count", 12)])

    assert passing.passed is True
    assert failing.passed is False


def test_guardrail_correctness_checks_refusals_and_legitimate_answers() -> None:
    assert guardrail_correctness(
        MEDICAL_RESPONSE,
        should_refuse=True,
        expected_category=GuardrailCategory.MEDICAL,
    ).passed
    assert guardrail_correctness(
        EATING_DISORDER_RESPONSE,
        should_refuse=True,
        expected_category=GuardrailCategory.EATING_DISORDER,
    ).passed
    assert guardrail_correctness("Use a controlled range of motion.", should_refuse=False).passed
    assert not guardrail_correctness(MEDICAL_RESPONSE, should_refuse=False).passed


def test_faithfulness_judge_validates_structured_result() -> None:
    judge = FaithfulnessJudge(
        RunnableLambda(
            lambda _: {"score": 5, "rationale": "Every claim is present in the evidence."}
        )
    )

    result = anyio.run(
        lambda: judge.evaluate(
            question="How did bench progress?",
            answer="It improved 17.86%.",
            evidence='{"percent_change": 17.86}',
        )
    )

    assert result == FaithfulnessJudgment(
        score=5,
        rationale="Every claim is present in the evidence.",
    )


def test_faithfulness_judge_scores_missing_evidence_without_model_call() -> None:
    judge = FaithfulnessJudge(
        RunnableLambda(lambda _: (_ for _ in ()).throw(AssertionError("must not run")))
    )

    result = anyio.run(
        lambda: judge.evaluate(question="Question", answer="Answer", evidence="  ")
    )

    assert result.score == 1
