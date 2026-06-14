import json
from pathlib import Path

import anyio
from langchain_core.runnables import RunnableLambda

from app.analysis.summary import build_analysis_summary
from app.eval.metrics import (
    CriteriaJudge,
    CriteriaJudgment,
    CriterionAssessment,
    FaithfulnessJudge,
    FaithfulnessJudgment,
    data_grounding,
    expected_data_points,
    guardrail_correctness,
    source_attribution,
    tool_selection,
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
    assert test_set.coverage_summary["mixed_units_kg_lb"] == ("A3",)
    assert test_set.agent_cases[2].expected_tool_calls[1].required is False


def test_analysis_case_expectations_match_sample_history() -> None:
    dataset = json.loads(Path("data/workout-history.json").read_text(encoding="utf-8"))
    test_set = load_test_set(Path("evaluation-test-set.json"))

    for case in test_set.analysis_cases:
        _, summary = build_analysis_summary(
            dataset["users"][case.user_id]["workouts"], case.question
        )
        expected = [(point.path, point.value) for point in case.expected_data_points]
        result = expected_data_points(summary, expected)
        assert result.passed, f"{case.case_id}: {result.details['mismatches']}"


def test_load_test_set_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "test-set.json"
    path.write_text(
        json.dumps(
            {
                "rag_cases": [
                    {
                        "case_id": "duplicate",
                        "question": "Question one?",
                        "edge_cases_covered": ["rag_edge"],
                        "expected_topic": "topic",
                        "expected_source_doc": "source.md",
                        "correct_answer_criteria": ["criterion"],
                        "failure_modes": ["failure"],
                    }
                ],
                "analysis_cases": [],
                "agent_cases": [],
                "guardrail_cases": [
                    {
                        "case_id": "duplicate",
                        "question": "Question two?",
                        "edge_cases_covered": ["safety_edge"],
                        "expected_category": "medical",
                        "correct_answer_criteria": ["criterion"],
                        "failure_modes": ["failure"],
                    }
                ],
                "coverage_summary": {
                    "rag_edge": ["duplicate"],
                    "safety_edge": ["duplicate"],
                },
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
                        "edge_cases_covered": ["not_blocked"],
                        "expected_category": "allowed",
                        "correct_answer_criteria": ["criterion"],
                        "failure_modes": ["failure"],
                    }
                ],
                "coverage_summary": {"not_blocked": ["not_blocked"]},
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


def test_load_test_set_rejects_unknown_coverage_case(tmp_path: Path) -> None:
    payload = json.loads(Path("evaluation-test-set.json").read_text(encoding="utf-8"))
    payload["coverage_summary"]["trend_detection"].append("unknown_case")
    path = tmp_path / "test-set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_test_set(path)
    except ValueError as exc:
        assert "references unknown cases: unknown_case" in str(exc)
    else:
        raise AssertionError("Unknown coverage case IDs must be rejected")


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


def test_tool_selection_supports_optional_tools_and_strict_order() -> None:
    passing = tool_selection(
        ["analyze_history", "rag_search"],
        ["analyze_history"],
        optional_tools=["rag_search"],
        strict_order=True,
    )
    wrong_order = tool_selection(
        ["rag_search", "analyze_history"],
        ["analyze_history"],
        optional_tools=["rag_search"],
        strict_order=True,
    )

    assert passing.passed is True
    assert wrong_order.passed is False


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


def test_criteria_judge_validates_detailed_result() -> None:
    judge = CriteriaJudge(
        RunnableLambda(
            lambda _: {
                "score": 4,
                "rationale": "The answer met the essential requirements.",
                "met_criteria": ["Uses normalized weights"],
                "missed_criteria": ["Could mention the exact switch date"],
                "observed_failure_modes": [],
                "criterion_assessments": [
                    {
                        "criterion": "Uses normalized weights",
                        "passed": True,
                        "answer_evidence": "The answer says 'after normalization'.",
                    }
                ],
            }
        )
    )

    result = anyio.run(
        lambda: judge.evaluate(
            question="How did bench progress?",
            answer="It rose from about 50 kg to 60 kg after normalization.",
            evidence='{"mixed_units_normalized": true}',
            expectations={"correct_answer_criteria": ["Uses normalized weights"]},
        )
    )

    assert result == CriteriaJudgment(
        score=4,
        rationale="The answer met the essential requirements.",
        met_criteria=["Uses normalized weights"],
        missed_criteria=["Could mention the exact switch date"],
        observed_failure_modes=[],
        criterion_assessments=[
            CriterionAssessment(
                criterion="Uses normalized weights",
                passed=True,
                answer_evidence="The answer says 'after normalization'.",
            )
        ],
    )
