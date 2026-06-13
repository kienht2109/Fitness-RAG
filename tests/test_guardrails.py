from typing import Any

import anyio
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag.guardrail_prompting import GuardrailClassification
from app.rag.guardrails import (
    EATING_DISORDER_RESPONSE,
    MEDICAL_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    GuardrailService,
    rule_based_decision,
)
from app.rag.models import GuardrailCategory


def test_rules_block_explicit_medical_diagnosis_without_model_call() -> None:
    decision = rule_based_decision("Do I Have a TORN Rotator Cuff?")

    assert decision is not None
    assert decision.category is GuardrailCategory.MEDICAL
    assert decision.response == MEDICAL_RESPONSE


def test_rules_block_dangerous_weight_control_behavior() -> None:
    decision = rule_based_decision("should i make myself vomit to lose weight?")

    assert decision is not None
    assert decision.category is GuardrailCategory.EATING_DISORDER
    assert decision.response == EATING_DISORDER_RESPONSE


def test_rules_do_not_block_general_fitness_questions() -> None:
    assert rule_based_decision("how can i prevent shoulder pain during bench press?") is None
    assert rule_based_decision("how large should a sustainable calorie deficit be?") is None


def test_classifier_blocks_out_of_scope_question() -> None:
    requests: list[Any] = []

    def classify(request: Any) -> GuardrailClassification:
        requests.append(request)
        return GuardrailClassification(category=GuardrailCategory.OUT_OF_SCOPE)

    service = GuardrailService(RunnableLambda(classify))
    decision = anyio.run(service.evaluate, "Explain how a database index works")

    assert requests
    assert decision.category is GuardrailCategory.OUT_OF_SCOPE
    assert decision.response == OUT_OF_SCOPE_RESPONSE


def test_rule_match_skips_classifier() -> None:
    def fail_if_called(_: Any) -> AIMessage:
        raise AssertionError("The classifier must not run after a deterministic match")

    service = GuardrailService(RunnableLambda(fail_if_called))
    decision = anyio.run(service.evaluate, "How do I treat my knee injury?")

    assert decision.category is GuardrailCategory.MEDICAL
