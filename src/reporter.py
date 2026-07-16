"""Summary reporter — generates an analysis report from labeling results."""

import csv
from collections import Counter
from pathlib import Path

from src.classifier import VALID_TYPES, is_valid_complaint_type


def generate_report(results_path: str = "output/results.csv") -> str:
    """Read the results CSV and produce a text summary report.

    Args:
        results_path: Path to the results CSV file.

    Returns:
        A formatted report string.

    """
    path = Path(results_path)
    if not path.exists():
        return "No results file found.\n"

    # Read and aggregate
    total = 0
    score_dist = Counter()
    keyword_counter = Counter()
    type_counter: Counter[str] = Counter()
    no_type = 0
    errors = 0
    score8plus = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

            # Sentiment score
            try:
                score = int(float(row.get("sentiment_score", -1)))
            except (ValueError, TypeError):
                score = -1
            if score == -1:
                errors += 1
            else:
                score_dist[score] += 1
                if score >= 8:
                    score8plus += 1

            # Complaint type
            ctype = (row.get("complaint_type") or "").strip()
            if ctype and is_valid_complaint_type(ctype):
                type_counter[ctype] += 1
            elif ctype:
                # Unknown type — still count
                type_counter[ctype] += 1
            else:
                no_type += 1

            # Keywords
            kw = (row.get("keywords") or "").strip()
            for k in kw.split(";") if kw else []:
                k = k.strip()
                if k:
                    keyword_counter[k] += 1

    # Build report
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  Customer Complaint Labeling — Summary Report")
    lines.append("=" * 60)
    lines.append(f"  Total records:         {total}")
    lines.append(f"  Processing errors:     {errors}")
    sev = (
        f"  Severe complaints (8+): {score8plus}"
        f" ({score8plus / max(total, 1) * 100:.1f}%)"
    )
    lines.append(sev)

    # Sentiment distribution
    lines.append("  Sentiment Score Distribution:")
    for s in [0, 2, 5, 8, 10]:
        cnt = score_dist.get(s, 0)
        bar = "█" * max(1, cnt // 3)
        pct = cnt / max(total, 1) * 100
        label = ""
        if s == 0:
            label = "Satisfied"
        elif s == 2:
            label = "Neutral"
        elif s == 5:
            label = "Slightly Dissatisfied"
        elif s == 8:
            label = "Complaint"
        elif s == 10:
            label = "Extreme Anger"
        lines.append(f"    {s:>2} ({label:22s}): {cnt:>3} ({pct:5.1f}%) {bar}")
    lines.append("")

    # Complaint type distribution
    lines.append("  Complaint Type Distribution:")
    known_types = type_counter.copy()
    for t in VALID_TYPES:
        known_types[t] = known_types.get(t, 0)
    if no_type > 0:
        known_types["(unclassified)"] = no_type
    for tp, cnt in known_types.most_common():
        bar = "█" * max(1, cnt // 2)
        lines.append(
            f"    {tp:30s}: {cnt:>3} ({cnt / max(total, 1) * 100:5.1f}%) {bar}"
        )
    lines.append("")

    # Top keywords
    lines.append("  Top Keywords:")
    for kw, cnt in keyword_counter.most_common(15):
        lines.append(f"    {kw:25s}: {cnt}")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def write_report(results_path: str, output_path: str = "output/report.txt") -> str:
    """Generate and save the summary report.

    Args:
        results_path: Path to the results CSV.
        output_path: Where to save the report.

    Returns:
        The report text.

    """
    report = generate_report(results_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    return report
