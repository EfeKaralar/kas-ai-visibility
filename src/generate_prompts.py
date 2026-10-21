"""Generate the traveler prompts from config/personas_attributes.yaml.

One helper-LLM call per base prompt writes all four language versions together, so they
share the same facts but are phrased naturally rather than translated. The LLM writes a
{budget} placeholder; amounts are rendered afterwards (€ for en/de/ru, TL at eur_try_rate
for tr), so changing the rate needs no LLM calls.

    uv run python -m src.generate_prompts

Outputs: data/processed/prompts.parquet (one row per prompt x language) and
data/review/prompts_review.md for human review before any engine call.
"""

import pandas as pd
from pydantic import BaseModel, Field

from src.common import PROCESSED, ROOT, SETTINGS, Budget, load_yaml
from src.llm import model_name, structured

STAGE = "prompt_generation"
PLACEHOLDER = "{budget}"
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish", "de": "German", "ru": "Russian"}

INSTRUCTIONS = """You write realistic messages that travelers type into an AI assistant such as ChatGPT
when looking for somewhere to stay. You will get a traveler profile, trip details and their needs.
Write one message per language.

Rules for every version:
- First person, as the traveler. Mention Kaş by name.
- Include every trip detail: month, number of nights, who is travelling, and the nightly budget.
- Write the budget only as the literal placeholder {budget}, exactly once, standing for the
  amount per room per night. Do not write any number or currency for the budget yourself.
- Mention every listed need and no other needs or preferences.
- Ask for specific places to stay. Do not name any hotel or area of Kaş yourself.
- Do not mention AI, prompts or this task.
- Write the town's name the way each language normally does: Kaş in English, Turkish and German,
  Каш in Russian.
- Give the traveler one concrete identity: write "my wife" or "my husband", never alternatives
  joined with a slash such as "husband/wife". Keep grammatical gender consistent throughout.

Languages: write each version the way a native speaker from that country would naturally write
it, with their own idioms and word order. Do not translate from the English version; keep the
same facts and needs. English: a British or American traveler. Turkish: a traveler from
Istanbul or Ankara. German: a traveler from Germany. Russian: a traveler from Russia."""


class PromptVersions(BaseModel):
    en: str = Field(description="English version")
    tr: str = Field(description="Turkish version")
    de: str = Field(description="German version")
    ru: str = Field(description="Russian version")


def brief(persona: dict, needs: list[str], style: str) -> str:
    return (f"Traveler: {persona['description']}.\n"
            f"Who is travelling: {persona['party']}.\n"
            f"Month: {persona['month']}. Nights: {persona['nights']}.\n"
            f"Needs: {'; '.join(needs)}.\n"
            f"Tone: {style}.")


def render_budget(language: str, euros: int, eur_try_rate: float) -> str:
    if language == "tr":
        lira = round(euros * eur_try_rate / 100) * 100
        return f"{lira:,}".replace(",", ".") + " TL"
    return f"€{euros}" if language == "en" else f"{euros} €"


def generate_versions(persona: dict, needs: list[str], style: str, budget: Budget) -> PromptVersions:
    messages = [("system", INSTRUCTIONS), ("human", brief(persona, needs, style))]
    for _ in range(2):
        versions = structured("helper", PromptVersions, messages, STAGE, budget)
        bad = [lang for lang, text in versions.model_dump().items() if text.count(PLACEHOLDER) != 1]
        if not bad:
            return versions
        messages = messages + [("human", f"The {', '.join(bad)} version(s) must contain {PLACEHOLDER} exactly "
                                         "once. Rewrite all versions following every rule.")]
    raise RuntimeError(f"Placeholder rule still broken for {persona['description']}: {bad}")


def write_review(prompts: pd.DataFrame, path) -> None:
    lines = [f"# Prompts for review ({prompts.base_id.nunique()} base prompts x "
             f"{prompts.language.nunique()} languages)", "",
             f"Generated with `{model_name('helper')}`. Budgets rendered per language. "
             "Check that each version is natural, matches the persona, and asks for every listed need.", ""]
    for base_id, group in prompts.groupby("base_id", sort=False):
        first = group.iloc[0]
        lines += [f"## {base_id} - {first.persona}",
                  f"*Needs:* {', '.join(first.attributes)} · *Budget:* €{first.budget_eur} · "
                  f"*Style:* {first.style}", ""]
        lines += [f"- **{row.language.upper()}**: {row.text}" for row in group.itertuples()]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    config = load_yaml("personas_attributes.yaml")
    settings, budget = SETTINGS["prompt_generation"], Budget()
    styles = settings["styles"]
    rows, index = [], 0
    for persona_id, persona in config["personas"].items():
        euros = config["budget_bands_eur"][persona["budget_band"]]
        for number, attribute_ids in enumerate(persona["prompts"], start=1):
            needs = [config["attributes"][a]["label"] for a in attribute_ids]
            style = styles[index % len(styles)]
            versions = generate_versions(persona, needs, style, budget).model_dump()
            base_id = f"{persona_id}_{number}"
            for language in settings["languages"]:
                amount = render_budget(language, euros, config["eur_try_rate"])
                rows.append({
                    "prompt_id": f"{base_id}_{language}", "base_id": base_id, "persona": persona_id,
                    "attributes": attribute_ids, "language": language, "style": style,
                    "budget_band": persona["budget_band"], "budget_eur": euros, "budget_display": amount,
                    "template": versions[language], "text": versions[language].replace(PLACEHOLDER, amount),
                })
            index += 1
            print(f"  {base_id}: done")

    prompts = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prompts.to_parquet(PROCESSED / "prompts.parquet", index=False)
    review = ROOT / "data" / "review" / "prompts_review.md"
    write_review(prompts, review)
    print(f"\n{prompts.base_id.nunique()} base prompts, {len(prompts)} prompts. Review: {review}")
    print(f"Spend so far: ${budget.spent():.4f}")


if __name__ == "__main__":
    main()
