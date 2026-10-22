"""Hand-check of the matcher (docs/metrics.md, "Matcher quality").

    uv run python -m src.hand_check prepare     # sample 30 answers, write the review files
    uv run python -m src.hand_check score       # precision and recall from the filled sheet

prepare: 30 main-run answers stratified by engine x language (12 groups, 2 or 3 each),
fixed seed. Writes data/review/hand_check_answers.md (answers to read) and
data/review/hand_check.csv (one row per extracted place, to mark). Refuses to overwrite a
sheet that already has marks.

score: precision = correct matched rows / matched rows; recall = correct matched rows /
(correct matched rows + missed places). Both with 95% Wilson intervals.
"""

import argparse
import math
import random
import sys

import pandas as pd

from src.common import PROCESSED, ROOT

SEED = 2026
SAMPLE_SIZE = 30
REVIEW = ROOT / "data" / "review"
SHEET = REVIEW / "hand_check.csv"
ANSWERS_FILE = REVIEW / "hand_check_answers.md"
MARK_COLUMNS = ["correct", "correct_place_if_wrong", "correct_label_if_wrong", "notes"]

HOW_TO = """# Hand-check: 30 answers

For each answer below, check the rows for that answer in `hand_check.csv`:

1. **correct** - `y` if the row is right: the place is the one the answer names AND the label is right
   (recommended = offered as an option; negative = to avoid / does not fit; incidental = landmark or
   passing reference). For an unmatched row (no matched place), `y` means it is truly not in our list.
   Otherwise `n`, and fill `correct_place_if_wrong` and/or `correct_label_if_wrong`.
2. **Missed places** - if the answer names a place to stay that has no row, add a row with the
   answer_id, the name in `name_in_answer`, and `missed` in `correct`.
3. Leave `notes` for anything unclear. Restaurants, beaches and areas are not places to stay.
"""


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (centre - half, centre + half)


def sample_answers(answers: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    answers = answers[answers.error.isna()].assign(language=lambda df: df.prompt_id.str[-2:])
    groups = sorted(answers.groupby(["engine", "language"]).groups)
    rng.shuffle(groups)
    sizes = {g: SAMPLE_SIZE // len(groups) + (1 if i < SAMPLE_SIZE % len(groups) else 0)
             for i, g in enumerate(groups)}
    picked = []
    for (engine, language), size in sorted(sizes.items()):
        ids = sorted(answers[(answers.engine == engine) & (answers.language == language)].answer_id)
        picked += rng.sample(ids, size)
    return answers[answers.answer_id.isin(picked)].sort_values(["engine", "language", "answer_id"])


def prepare():
    if SHEET.exists() and pd.read_csv(SHEET)["correct"].notna().any():
        sys.exit(f"{SHEET} already has marks; not overwriting.")
    answers = pd.read_parquet(PROCESSED / "answers.parquet")
    prompts = pd.read_parquet(PROCESSED / "prompts.parquet").set_index("prompt_id").text
    mentions = pd.read_parquet(PROCESSED / "mentions.parquet")
    sample = sample_answers(answers)

    sheet = (mentions[mentions.answer_id.isin(sample.answer_id)]
             .assign(matched_place=lambda df: df["name"])
             [["answer_id", "order", "name_in_answer", "matched_place", "place_uid", "label", "matched_by",
               "needs_check"]]
             .sort_values(["answer_id", "order"]))
    for column in MARK_COLUMNS:
        sheet[column] = ""
    REVIEW.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET, index=False)

    lines = [HOW_TO]
    for number, answer in enumerate(sample.itertuples(), start=1):
        lines += [f"## {number}. {answer.answer_id}", f"**Prompt:** {prompts[answer.prompt_id]}", "",
                  answer.text, "", "---", ""]
    ANSWERS_FILE.write_text("\n".join(lines))
    print(f"{len(sample)} answers, {len(sheet)} rows to mark")
    print(sample.groupby(["engine", "language"]).size().unstack().to_string())
    print(f"Read: {ANSWERS_FILE}\nMark: {SHEET}")


def score():
    sheet = pd.read_csv(SHEET, dtype=str)
    marks = sheet["correct"].str.strip().str.lower()
    unmarked = sheet[marks.isna() | (marks == "")]
    if len(unmarked):
        sys.exit(f"{len(unmarked)} rows are not marked yet.")
    matched = sheet.place_uid.notna() & (sheet.place_uid != "")
    correct_matched = int((matched & (marks == "y")).sum())
    total_matched = int(matched.sum())
    # missed: places with no row, plus unmatched rows that should have matched a listed place
    missed = int((marks == "missed").sum() + (~matched & (marks == "n")).sum())
    precision = correct_matched / total_matched
    recall = correct_matched / (correct_matched + missed)
    p_low, p_high = wilson(correct_matched, total_matched)
    r_low, r_high = wilson(correct_matched, correct_matched + missed)
    result = pd.DataFrame([
        {"metric": "precision", "value": precision, "low_95": p_low, "high_95": p_high,
         "numerator": correct_matched, "denominator": total_matched},
        {"metric": "recall", "value": recall, "low_95": r_low, "high_95": r_high,
         "numerator": correct_matched, "denominator": correct_matched + missed}])
    folder = PROCESSED / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    result.to_parquet(folder / "matcher_quality.parquet", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "score"])
    {"prepare": prepare, "score": score}[parser.parse_args().action]()
