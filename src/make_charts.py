"""Export the six slide charts (PNG) from the metric tables.

    uv run python -m src.make_charts                       # data/processed/metrics -> docs/charts
    uv run python -m src.make_charts --metrics DIR --out DIR

Colours follow the validated reference palette (dataviz skill): engines use the first three
categorical slots (safe for colour-vision deficiency across all pairs); source categories use
a fixed slot order so a category keeps its colour on every chart; "other" and "no source"
are neutral grey. Text never takes a series colour.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PROCESSED, ROOT  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
ENGINE_COLORS = {"openai": "#2a78d6", "gemini": "#eb6834", "anthropic": "#1baf7a"}
ENGINE_NAMES = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Anthropic"}
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish", "de": "German", "ru": "Russian"}
CATEGORY_ORDER = ["ota", "review_site", "own_website", "map", "blog_media", "other", "no_source"]
CATEGORY_COLORS = {"ota": "#2a78d6", "review_site": "#eb6834", "own_website": "#1baf7a", "map": "#eda100",
                   "blog_media": "#e87ba4", "other": "#bdbcb6", "no_source": "#e4e3df"}
CATEGORY_NAMES = {"ota": "OTA", "review_site": "Review site", "own_website": "Hotel's own website", "map": "Map",
                  "blog_media": "Blog / media", "other": "Other", "no_source": "No source attached"}
TARGET_NAMES = {"listed_outside_destination": "Outside Kaş", "outside_destination": "Outside Kaş",
                "not_in_place_list": "In Kaş, not in our list", "permanently_closed": "Permanently closed",
                "osm_only": "Only in OpenStreetMap", "unverifiable": "Unverifiable",
                "not_accommodation": "Not accommodation", "listed_not_accommodation": "Not accommodation",
                "in_list_missed_by_matcher": "Matcher miss", "unclassified": "Unclassified",
                "not_looked_up": "Not looked up (named once)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def percent_axis(axis, ax):
    formatter = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def finish(fig, path: Path, note: str):
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5, ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def coverage_chart(metrics: Path, out: Path):
    table = pd.read_parquet(metrics / "coverage.parquet")
    rows = [("All engines", "#52514e", table[table.scope == "overall"].iloc[0])]
    rows += [(ENGINE_NAMES[engine], ENGINE_COLORS[engine], row)
             for engine, row in table[table.scope == "engine"].set_index("group").iterrows()]
    rows = rows[::-1]                                     # barh draws bottom-up
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh([r[0] for r in rows], [r[2].coverage for r in rows], color=[r[1] for r in rows], height=0.55)
    for bar, (_, _, row) in zip(bars, rows):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row.coverage:.0%}  ({row.places_recommended} of {row.destination_places})",
                va="center", color=INK_2, fontsize=10)
    ax.set_xlim(0, max(r[2].coverage for r in rows) * 1.35 or 1)
    percent_axis("x", ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of Kaş places recommended at least once")
    finish(fig, out / "1_coverage_by_engine.png", "Coverage = places named as a recommendation in ≥1 answer "
           "/ all accommodation in the Kaş boundary.")


def concentration_chart(metrics: Path, out: Path):
    summary = pd.read_parquet(metrics / "place_summary.parquet")
    overall = pd.read_parquet(metrics / "concentration.parquet").query("scope == 'overall'").iloc[0]
    top_n, shown = int(overall.top_n), 15
    ranked = summary[summary.recommendations > 0].sort_values("recommendations", ascending=False)
    head = ranked.head(shown).iloc[::-1]                  # barh draws bottom-up
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#2a78d6" if rank < top_n else "#b7d3f6" for rank in range(len(head))][::-1]
    bars = ax.barh(head.name, head.recommendations, color=colors, height=0.65)
    for bar, value in zip(bars, head.recommendations):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, str(value), va="center",
                fontsize=9, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Recommendations")
    ax.set_title(f"Top {top_n} places receive {overall.top_n_share:.0%} of all recommendations")
    finish(fig, out / "2_concentration.png", f"Top {min(shown, len(ranked))} of {len(ranked)} places recommended "
           f"at least once; {len(summary) - len(ranked)} of {len(summary)} Kaş places were never recommended.")


def popularity_chart(metrics: Path, out: Path):
    summary = pd.read_parquet(metrics / "place_summary.parquet").dropna(subset=["review_count"])
    rho = pd.read_parquet(metrics / "popularity.parquet").spearman_rho.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    never = summary[summary.recommendations == 0]
    some = summary[summary.recommendations > 0]
    ax.scatter(never.review_count.clip(lower=1), never.recommendations, s=14, color="#bdbcb6",
               label="Never recommended", linewidths=0)
    ax.scatter(some.review_count.clip(lower=1), some.recommendations, s=22, color="#2a78d6",
               edgecolors=SURFACE, linewidths=0.8, label="Recommended")
    for row in some.nlargest(6, "recommendations").itertuples():
        ax.annotate(row.name, (max(row.review_count, 1), row.recommendations), textcoords="offset points",
                    xytext=(5, 3), fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Google review count (log scale)")
    ax.set_ylabel("Recommendations")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Popularity and AI visibility (Spearman ρ = {rho:.2f})")
    finish(fig, out / "3_popularity_scatter.png", "Each dot is a Kaş place with Google reviews. "
           "Correlation describes a pattern; it does not show cause.")


def sources_chart(metrics: Path, out: Path):
    table = pd.read_parquet(metrics / "sources_per_recommendation.parquet")
    pairs = table[table.category != "no_source"]
    groups = [("overall", "all", "All")] + [("engine", e, ENGINE_NAMES[e]) for e in ENGINE_NAMES] + \
             [("language", lang, LANGUAGE_NAMES[lang]) for lang in LANGUAGE_NAMES]
    groups = [g for g in groups if ((pairs.scope == g[0]) & (pairs.group == g[1])).any()]
    categories = [c for c in CATEGORY_ORDER if c in set(pairs.category)]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [g[2] for g in groups][::-1]
    left = [0.0] * len(groups)
    for category in categories:
        values = []
        for scope, group, _ in groups[::-1]:
            match = pairs[(pairs.scope == scope) & (pairs.group == group) & (pairs.category == category)]
            values.append(float(match.share_of_pairs.iloc[0]) if len(match) else 0.0)
        ax.barh(labels, values, left=left, color=CATEGORY_COLORS[category], height=0.6,
                edgecolor=SURFACE, linewidth=1.5, label=CATEGORY_NAMES[category])
        for i, value in enumerate(values):
            if value >= 0.08:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=8.5, color=INK)
        left = [a + b for a, b in zip(left, values)]
    percent_axis("x", ax)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_title("What sources back each recommended place", pad=48)
    finish(fig, out / "4_sources_by_engine_language.png", "Share of (recommended place, cited domain) pairs; "
           "a domain counts once per recommendation.")


def hypothesis_chart(metrics: Path, out: Path):
    table = pd.read_parquet(metrics / "hypothesis.parquet")
    verdict = table.verdict.iloc[0]
    groups = ["all"] + [e for e in ENGINE_NAMES if e in set(table.group)]
    names = ["All engines"] + [ENGINE_NAMES[e] for e in groups[1:]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.36
    for offset, column, label, color in ((-width / 2, "ota", "Backed by an OTA", "#2a78d6"),
                                         (width / 2, "own_website", "Backed by the hotel's own website", "#1baf7a")):
        values = [table.set_index("group").at[g, column] for g in groups]
        bars = ax.bar([i + offset for i in range(len(groups))], values, width=width - 0.04, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.0%}", ha="center", va="bottom",
                    fontsize=9.5, color=INK_2)
    ax.set_xticks(range(len(groups)), names)
    percent_axis("y", ax)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Who backs a recommended Kaş place: OTAs or its own website?", pad=24)
    ax.text(0, 1.02, f"Hypothesis (OTA-backed more often than own website): {verdict}", transform=ax.transAxes,
            fontsize=10, color=INK_2)
    finish(fig, out / "5_own_website_vs_ota.png", "Recommended places that have a website. A recommendation can "
           "be backed by both. Decision rule fixed in advance in docs/metrics.md.")


def targets_chart(metrics: Path, out: Path):
    table = pd.read_parquet(metrics / "recommendation_targets.parquet").query("engine == 'all'")
    table = table[table.target != "destination"].copy()
    table["label"] = table.target.map(TARGET_NAMES).fillna(table.target)
    table = table.groupby("label", as_index=False)[["recommendations", "share"]].sum().sort_values("recommendations")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(table.label, table.recommendations, color="#eb6834", height=0.55)
    for bar, row in zip(bars, table.itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row.recommendations}  ({row.share:.1%} of all)" if row.share < 0.01
                else f"{row.recommendations}  ({row.share:.0%} of all)", va="center", fontsize=9.5, color=INK_2)
    ax.set_xlim(0, table.recommendations.max() * 1.35 if len(table) else 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Recommendations")
    ax.set_title("Recommendations that are not a Kaş place in our list")
    finish(fig, out / "6_outside_the_list.png", "Percentages are of all recommendations. Closures and unverifiable "
           "names were checked by hand where noted in config/manual_verifications.yaml.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=PROCESSED / "metrics")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "charts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for chart in (coverage_chart, concentration_chart, popularity_chart, sources_chart, hypothesis_chart,
                  targets_chart):
        chart(args.metrics, args.out)
    print(f"Charts written to {args.out}")


if __name__ == "__main__":
    main()
