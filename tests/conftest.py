import asyncio
from typing import TypeVar

import pytest
from pydantic import BaseModel

from fineprint.llm import Effort
from fineprint.schemas import Analysis, KeyTerm, Obligation, Quote, Risk

T = TypeVar("T", bound=BaseModel)

LEASE = """\
1. Rent. The Tenant shall pay £900 per month on the first day of each month.
2. Deposit. A deposit of £1,800 is due on signing. The deposit is non-refundable.
3. Termination. Either party may terminate this agreement on sixty (60) days' written notice.
4. Renewal. This agreement renews automatically for successive twelve-month terms."""


class FakeLLM:
    """Stands in for the model: returns a canned result or raises, and records each call."""

    def __init__(self, result: BaseModel) -> None:
        self.result = result
        self.error: Exception | None = None
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        *,
        system: str,
        documents: dict[str, str],
        request: str,
        output_model: type[T],
        effort: Effort = "default",
    ) -> T:
        await asyncio.sleep(0)  # a real model call suspends; concurrency tests rely on it
        self.calls.append(
            {
                "system": system,
                "documents": documents,
                "request": request,
                "output_model": output_model,
                "effort": effort,
            }
        )
        if self.error:
            raise self.error
        assert isinstance(self.result, output_model), "the assistant asked for the wrong schema"
        return self.result.model_copy(deep=True)


def _quote(text: str) -> Quote:
    return Quote(document="A", text=text)


@pytest.fixture
def lease() -> str:
    return LEASE


@pytest.fixture
def analysis() -> Analysis:
    return Analysis(
        perspective="the Tenant",
        is_legal_document=True,
        document_type="Residential lease",
        summary="A monthly lease with a non-refundable deposit that renews automatically.",
        parties=["Landlord", "Tenant"],
        key_terms=[
            KeyTerm(label="Rent", value="£900 a month", quote=_quote("shall pay £900 per month"))
        ],
        obligations=[
            Obligation(
                party="Tenant",
                description="Pay rent on the 1st of each month.",
                when="Monthly, on the 1st",
                quote=_quote("on the first day of each month"),
            )
        ],
        risks=[
            Risk(
                title="Automatic renewal",
                plain_language="The lease renews for another year unless ended.",
                why_it_matters="You could be locked in for another twelve months.",
                severity="medium",
                quotes=[_quote("This agreement renews automatically")],
            ),
            Risk(
                title="Deposit is non-refundable",
                plain_language="You never get the £1,800 deposit back.",
                why_it_matters="Deposits are usually refundable minus damage.",
                severity="high",
                quotes=[_quote("The deposit is non-refundable.")],
            ),
        ],
        inconsistencies=[],
        missing_or_unclear=["Who pays for repairs"],
        questions_for_lawyer=["Is the non-refundable deposit in clause 2 enforceable?"],
        next_steps=["Ask the landlord to make the deposit refundable, in writing."],
    )


@pytest.fixture
def llm(analysis: Analysis) -> FakeLLM:
    return FakeLLM(analysis)
