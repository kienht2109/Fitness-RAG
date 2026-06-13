from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.rag.guardrail_prompting import GuardrailClassification, build_guardrail_classifier
from app.rag.models import GuardrailCategory, GuardrailDecision


OUT_OF_SCOPE_RESPONSE = (
    "I can only help with fitness, exercise, training, nutrition basics, and recovery questions."
)
MEDICAL_RESPONSE = (
    "I'm sorry you're dealing with this. I can't diagnose an injury or recommend treatment. "
    "Please consult a qualified doctor or physical therapist for an assessment. If your symptoms "
    "are severe or rapidly worsening, seek urgent medical care."
)
EATING_DISORDER_RESPONSE = (
    "I'm sorry you're dealing with this. I can't help with purging, starvation, extreme food "
    "restriction, or other dangerous weight-control behaviors. Please contact a qualified doctor "
    "or registered dietitian for support. If you're in immediate danger, contact local emergency "
    "services."
)

_EATING_DISORDER_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:make|force) myself (?:vomit|throw up)\b",
        r"\bpurge(?:ing)?\b",
        r"\b(?:use|take|abuse) laxatives? (?:to|for) (?:lose|weight)\b",
        r"\bstarve myself\b",
        r"\b(?:stop|avoid) eating\b.*\b(?:lose weight|thin|skinny)\b",
        r"\b(?:[1-7]\d{2}) calories? (?:a|per) day\b",
        r"\b(?:pro[ -]?ana|thinspo)\b",
    )
)

_DIRECT_MEDICAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bdiagnos(?:e|is|ing)\b",
        r"\b(?:prescribe|prescription)\b",
    )
)

_MEDICAL_INTENT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:do|could|might) i have\b",
        r"\bwhat(?:'s| is) wrong with\b",
        r"\b(?:is|could) (?:this|it) (?:a|an)\b",
        r"\bhow (?:do|should|can) i treat\b",
        r"\b(?:treatment|remedy|cure) for\b",
        r"\bshould i (?:see|visit|consult) (?:a |an )?(?:doctor|physician|physio|therapist)\b",
        r"\bis (?:this |my )?.{0,30}\bnormal\b",
    )
)

_MEDICAL_SIGNAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:pain|hurt|injur(?:y|ed)|tear|torn|sprain|strain|fracture|broken)\b",
        r"\b(?:swelling|numbness|tingling|weakness|dislocat(?:ed|ion))\b",
        r"\b(?:tendonitis|tendinitis|impingement|inflammation|symptoms?|condition)\b",
        r"\b(?:knee|shoulder|back|spine|neck|hip|ankle|wrist|elbow|joint|muscle|tendon)\b",
    )
)

_OUT_OF_SCOPE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bwhat(?:'s| is) the weather\b",
        r"\bweather forecast\b",
        r"\b(?:write|debug) (?:some |a )?(?:code|python|javascript)\b",
        r"\b(?:stock|share|crypto) price\b",
        r"\bwho (?:is|won) (?:the )?(?:president|election)\b",
        r"\b(?:book|find) (?:a )?(?:flight|hotel)\b",
    )
)

_RESPONSES = {
    GuardrailCategory.OUT_OF_SCOPE: OUT_OF_SCOPE_RESPONSE,
    GuardrailCategory.MEDICAL: MEDICAL_RESPONSE,
    GuardrailCategory.EATING_DISORDER: EATING_DISORDER_RESPONSE,
}


class GuardrailService:
    def __init__(self, classifier: Runnable[Any, GuardrailClassification]) -> None:
        self.classifier = classifier

    async def evaluate(self, question: str) -> GuardrailDecision:
        rule_decision = rule_based_decision(question)
        if rule_decision is not None:
            return rule_decision

        classification = await self.classifier.ainvoke({"question": question})
        return decision_for(classification.category)


def rule_based_decision(question: str) -> GuardrailDecision | None:
    question = " ".join(question.lower().split())
    if _matches_any(_EATING_DISORDER_PATTERNS, question):
        return decision_for(GuardrailCategory.EATING_DISORDER)
    if _matches_any(_DIRECT_MEDICAL_PATTERNS, question):
        return decision_for(GuardrailCategory.MEDICAL)
    if _matches_any(_MEDICAL_INTENT_PATTERNS, question) and _matches_any(
        _MEDICAL_SIGNAL_PATTERNS, question
    ):
        return decision_for(GuardrailCategory.MEDICAL)
    if _matches_any(_OUT_OF_SCOPE_PATTERNS, question):
        return decision_for(GuardrailCategory.OUT_OF_SCOPE)
    return None


def decision_for(category: GuardrailCategory) -> GuardrailDecision:
    return GuardrailDecision(category=category, response=_RESPONSES.get(category))


def create_guardrail_service(chat_model: ChatOpenAI) -> GuardrailService:
    return GuardrailService(build_guardrail_classifier(chat_model))


def _matches_any(patterns: tuple[re.Pattern[str], ...], question: str) -> bool:
    return any(pattern.search(question) for pattern in patterns)
