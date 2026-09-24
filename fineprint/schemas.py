"""Request, response and model-output shapes.

Output models are sent to the model as its response schema, so they stay plain: every field
is required and there are no length or range constraints (the model does not enforce them).
Field descriptions travel with the schema and act as per-field instructions.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .documents import MAX_DOCUMENT_CHARS

Task = Literal["analyze", "ask", "compare"]
Severity = Literal["low", "medium", "high"]
DocumentLabel = Literal["A", "B"]


class Quote(BaseModel):
    document: DocumentLabel = Field(description="The document the passage is copied from.")
    text: str = Field(
        description="One passage copied exactly from that document, ideally under 40 words."
    )


class KeyTerm(BaseModel):
    label: str = Field(description="For example: Rent, Start date, Notice period.")
    value: str
    quote: Quote


class Obligation(BaseModel):
    party: str
    description: str = Field(description="What this party must or must not do, in plain words.")
    when: str = Field(description="The deadline or trigger, e.g. 'Within 14 days', or 'Ongoing'.")
    quote: Quote


class Risk(BaseModel):
    title: str
    plain_language: str = Field(description="What the quoted text says, in plain words.")
    why_it_matters: str = Field(description="The concrete consequence for the reader.")
    severity: Severity
    quotes: list[Quote] = Field(
        description="1-3 passages: the clause plus any definition, exception or schedule "
        "it depends on."
    )


class Analysis(BaseModel):
    perspective: str = Field(description="The party this is written for, e.g. 'the Tenant'.")
    is_legal_document: bool
    document_type: str
    summary: str = Field(description="Three to five plain sentences.")
    parties: list[str]
    key_terms: list[KeyTerm]
    obligations: list[Obligation]
    risks: list[Risk]
    inconsistencies: list[str] = Field(
        description="Clauses that contradict each other, naming both clauses."
    )
    missing_or_unclear: list[str]
    questions_for_lawyer: list[str] = Field(
        description="Each names the clause and what the reader needs to find out."
    )
    next_steps: list[str] = Field(
        description="Concrete actions in order, including the choices open to the reader."
    )


class Answer(BaseModel):
    answer: str
    supported_by_document: Literal["yes", "partly", "no"]
    quotes: list[Quote]
    caveats: list[str] = Field(description="Conditions, exceptions or gaps that affect the answer.")
    next_step: str = Field(
        description="The one most useful thing to do next, or a question for a lawyer."
    )


class Difference(BaseModel):
    topic: str
    document_a: str = Field(description="What Document A says, or 'Not addressed'.")
    document_b: str = Field(description="What Document B says, or 'Not addressed'.")
    impact: str = Field(description="What the difference means for the reader.")
    severity: Severity
    quotes: list[Quote]


class Comparison(BaseModel):
    perspective: str = Field(description="The party this is written for, e.g. 'the Freelancer'.")
    is_legal_document: bool = Field(description="Whether both texts are legal documents.")
    summary: str
    differences: list[Difference]
    questions_for_lawyer: list[str]
    next_steps: list[str] = Field(
        description="Concrete actions in order, e.g. which version to prefer or what to negotiate."
    )


# The three shapes the model can return, one per task.
Result = Analysis | Answer | Comparison


DocumentText = Annotated[str, Field(max_length=MAX_DOCUMENT_CHARS)]


class AssistRequest(BaseModel):
    """What the user submits: one or two documents, plus an optional situation and question."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentText] = Field(min_length=1, max_length=2)
    question: str | None = Field(default=None, max_length=1000)
    context: str | None = Field(default=None, max_length=500)

    @field_validator("documents")
    @classmethod
    def _documents_have_text(cls, documents: list[str]) -> list[str]:
        if any(not document.strip() for document in documents):
            raise ValueError("each document must contain some text")
        return documents

    @field_validator("question", "context")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class AssistResponse(BaseModel):
    task: Task
    result: Result
    warnings: list[str]
    disclaimer: str
