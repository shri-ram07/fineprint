import asyncio

import pytest
from pydantic import ValidationError

from fineprint.assistant import assist, resolve_task, verify_quotes
from fineprint.documents import MAX_DOCUMENT_CHARS
from fineprint.llm import LLMError
from fineprint.schemas import Answer, AssistRequest, Comparison, Difference, Quote, Risk

PDF_LIKE = (
    "The Tenant shall pay £900 per month. Either party may termi-\nnate on sixty (60) "
    "days’ written notice. The ﬁnal inspection is due on exit. Late fees are 5% of rent."
)


def run(request: AssistRequest, llm) -> object:
    return asyncio.run(assist(request, llm))


def quote(text: str, document: str = "A") -> Quote:
    return Quote(document=document, text=text)


def answer_with(*quotes) -> Answer:
    return Answer(answer="...", supported_by_document="yes", quotes=list(quotes), caveats=[])


@pytest.mark.parametrize(
    ("documents", "question", "expected"),
    [
        (["lease"], None, "analyze"),
        (["lease"], "Can I sublet?", "ask"),
        (["v1", "v2"], None, "compare"),
        (["v1", "v2"], "Which has the longer notice period?", "ask"),
    ],
)
def test_resolve_task(documents, question, expected):
    assert resolve_task(AssistRequest(documents=documents, question=question)) == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"documents": []},
        {"documents": ["a", "b", "c"]},
        {"documents": ["   \n"]},
        {"documents": ["x" * (MAX_DOCUMENT_CHARS + 1)]},
        {"documents": ["lease"], "question": "?" * 1001},
        {"documents": ["lease"], "context": "x" * 501},
    ],
    ids=[
        "no-documents",
        "three-documents",
        "blank-document",
        "too-long",
        "long-question",
        "long-context",
    ],
)
def test_invalid_requests_are_rejected(fields):
    with pytest.raises(ValidationError):
        AssistRequest(**fields)


def test_blank_question_and_context_are_ignored():
    request = AssistRequest(documents=["lease"], question="  ", context="\n")
    assert (request.question, request.context) == (None, None)
    assert resolve_task(request) == "analyze"


@pytest.mark.parametrize(
    ("quoted", "expected"),
    [
        ("The Tenant shall pay £900 per month.", "The Tenant shall pay £900 per month"),
        (
            "Either party may terminate on sixty (60) days' written notice",
            "Either party may termi-\nnate on sixty (60) days’ written notice",
        ),
        ("the final inspection", "The ﬁnal inspection"),
        ("late fees are 5% of rent", "Late fees are 5% of rent"),
        ("The Tenant shall pay £950 per month.", ""),
        ("Late fees are 5 of rent", ""),
        ("...", ""),
        ("", ""),
    ],
    ids=[
        "exact-without-edge-punctuation",
        "hyphenation-and-curly-quote",
        "ligature-and-case",
        "percent-kept",
        "altered-amount",
        "dropped-percent",
        "punctuation-only",
        "empty",
    ],
)
def test_quotes_are_verified_and_rewritten_to_the_source(quoted, expected):
    answer = answer_with(quote(quoted))
    unmatched = verify_quotes(answer, {"A": PDF_LIKE})
    assert answer.quotes[0].text == expected
    assert unmatched == (0 if expected else 1)


def test_quote_must_come_from_the_document_it_names():
    answer = answer_with(
        quote("Rent is £1,000", document="A"), quote("Rent is £1,000", document="B")
    )
    assert verify_quotes(answer, {"A": "Rent is £900.", "B": "Rent is £1,000."}) == 1
    assert [q.text for q in answer.quotes] == ["", "Rent is £1,000"]


def test_quote_labelled_with_a_missing_document_is_unmatched():
    answer = answer_with(quote("Rent is £900", document="B"))
    assert verify_quotes(answer, {"A": "Rent is £900."}) == 1


def test_risks_without_any_quote_count_as_unmatched(analysis, lease):
    assert verify_quotes(analysis, {"A": lease}) == 0
    analysis.risks[0].quotes = []
    assert verify_quotes(analysis, {"A": lease}) == 1


def test_differences_without_any_quote_count_as_unmatched():
    comparison = Comparison(
        perspective="the Freelancer",
        summary="",
        differences=[
            Difference(
                topic="Payment",
                document_a="30 days",
                document_b="60 days",
                impact="",
                severity="high",
                quotes=[],
            )
        ],
        questions_for_lawyer=[],
    )
    assert verify_quotes(comparison, {"A": "x", "B": "y"}) == 1


def test_analyze_sends_the_document_and_situation_and_sorts_risks(llm, lease):
    response = run(AssistRequest(documents=[lease], context="I'm the tenant"), llm)

    call = llm.calls[0]
    assert call["documents"] == {"A": lease}
    assert "I'm the tenant" in call["request"]
    assert "Analyse Document A" in call["request"]

    assert response.task == "analyze"
    assert [risk.severity for risk in response.result.risks] == ["high", "medium"]
    assert response.warnings == []
    assert "not legal advice" in response.disclaimer


def test_compare_labels_both_documents_and_sorts_differences(llm, lease):
    differences = [
        Difference(
            topic=severity,
            document_a="",
            document_b="",
            impact="",
            severity=severity,
            quotes=[quote("The deposit is non-refundable.")],
        )
        for severity in ("low", "high")
    ]
    llm.result = Comparison(
        perspective="the Tenant", summary="", differences=differences, questions_for_lawyer=[]
    )
    response = run(AssistRequest(documents=[lease, "Rent is £950."]), llm)

    assert llm.calls[0]["documents"] == {"A": lease, "B": "Rent is £950."}
    assert response.task == "compare"
    assert [difference.severity for difference in response.result.differences] == ["high", "low"]


def test_ask_sends_the_question(llm, lease):
    llm.result = answer_with(quote("The deposit is non-refundable."))
    response = run(AssistRequest(documents=[lease], question="Do I get my deposit back?"), llm)

    assert "Do I get my deposit back?" in llm.calls[0]["request"]
    assert response.task == "ask"
    assert response.warnings == []


def test_unmatched_quotes_are_blanked_and_reported(llm, analysis, lease):
    analysis.risks.append(
        Risk(
            title="Invented",
            plain_language="",
            why_it_matters="",
            severity="low",
            quotes=[quote("The Landlord may enter at any time.")],
        )
    )
    llm.result = analysis
    response = run(AssistRequest(documents=[lease]), llm)

    invented = next(risk for risk in response.result.risks if risk.title == "Invented")
    assert invented.quotes[0].text == ""
    assert response.warnings[0].startswith("Some points (1) could not be matched")


def test_non_legal_document_is_flagged(llm, analysis, lease):
    llm.result = analysis.model_copy(update={"is_legal_document": False})
    response = run(AssistRequest(documents=[lease]), llm)
    assert any("doesn't look like a legal document" in warning for warning in response.warnings)


@pytest.mark.parametrize(("support", "flagged"), [("yes", True), ("partly", True), ("no", False)])
def test_answer_claiming_support_without_quotes_is_flagged(llm, lease, support, flagged):
    llm.result = Answer(answer="...", supported_by_document=support, quotes=[], caveats=[])
    response = run(AssistRequest(documents=[lease], question="Can I keep a pet?"), llm)
    assert any("cites no passage" in warning for warning in response.warnings) == flagged


def test_unmatched_answer_quote_is_reported_once(llm, lease):
    llm.result = answer_with(quote("You may keep a pet."))
    response = run(AssistRequest(documents=[lease], question="Can I keep a pet?"), llm)
    assert len(response.warnings) == 1


def test_llm_errors_propagate(llm, lease):
    llm.error = LLMError("unavailable")
    with pytest.raises(LLMError):
        run(AssistRequest(documents=[lease]), llm)
