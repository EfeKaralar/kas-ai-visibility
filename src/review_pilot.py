"""Label pilot answers to decide on trip-context additions and automatic follow-ups.

For each pilot answer the helper LLM records whether it recommends specific places, whether
it asks the traveler a clarifying question and about what, and whether it hedges because of
missing exact dates. Decision rules (docs/plan.md): a clarifying topic in >10% of answers is
added to the trip context; >20% of answers without a recommendation triggers follow-ups.

    uv run python -m src.review_pilot
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.common import PROCESSED, Budget
from src.llm import structured

STAGE = "pilot_review"
Topic = Literal["budget", "dates", "group", "location_within_destination", "amenities", "other"]


class PilotLabel(BaseModel):
    recommends_specific_places: bool = Field(description="names at least one specific hotel, pension or other "
                                                          "place to stay as an option for the traveler")
    asks_clarifying_question: bool = Field(description="asks the traveler for more information before or "
                                                       "instead of fully answering")
    clarifying_topics: list[Topic] = Field(description="what the clarifying questions are about; empty if none")
    hedges_on_dates: bool = Field(description="says prices or availability cannot be confirmed without exact dates")


INSTRUCTIONS = """You label answers that an AI assistant gave to a traveler asking where to stay.
Read the answer and fill in the fields exactly as described. Judge only what the answer says."""


def main():
    budget = Budget()
    answers = pd.read_parquet(PROCESSED / "answers_pilot.parquet")
    labels = []
    for answer in answers.itertuples():
        label = structured("helper", PilotLabel, [("system", INSTRUCTIONS), ("human", answer.text)], STAGE, budget)
        labels.append({"answer_id": answer.answer_id, **label.model_dump()})
    labelled = answers.merge(pd.DataFrame(labels), on="answer_id")
    labelled["language"] = labelled.prompt_id.str[-2:]
    labelled.to_parquet(PROCESSED / "pilot_labels.parquet", index=False)

    n = len(labelled)
    no_recommendation = (~labelled.recommends_specific_places).mean()
    print(f"Answers: {n}")
    print(f"No specific recommendation: {no_recommendation:.0%} (follow-ups if > 20%)")
    print(f"Asks a clarifying question: {labelled.asks_clarifying_question.mean():.0%}")
    topics = labelled.explode("clarifying_topics").clarifying_topics.dropna().value_counts() / n
    print("Clarifying topics, share of answers (add to trip context if > 10%):")
    print(topics.map("{:.0%}".format).to_string() if len(topics) else "  none")
    print(f"Hedges on dates: {labelled.hedges_on_dates.mean():.0%}")
    print("\nBy engine:")
    print(labelled.groupby("engine")[["recommends_specific_places", "asks_clarifying_question",
                                      "hedges_on_dates"]].mean().round(2).to_string())
    print("\nBy language:")
    print(labelled.groupby("language")[["recommends_specific_places", "asks_clarifying_question"]]
          .mean().round(2).to_string())
    print(f"\nSpend so far: ${budget.spent():.2f}")


if __name__ == "__main__":
    main()
