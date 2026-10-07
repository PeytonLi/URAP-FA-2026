#!/usr/bin/env python3
"""Generate a two-page core PDF plus a linked Markdown results report."""

import json
from collections import Counter
from html import escape

from common import ROOT, read_csv, write_csv
from analyze_evidence import NAMES, ORDER


def main():
    report = ROOT / "report"
    analysis = ROOT / "results/analysis"
    iat = {
        d: {r["model_key"]: r for r in read_csv(analysis / f"iat_{d}_significance.csv")}
        for d in ("neutral12", "utkface")
    }
    dt = read_csv(analysis / "decodingtrust_responses.csv")
    bold = read_csv(analysis / "bold_scored_responses.csv")
    vis = {
        r["model_key"]: r for r in read_csv(analysis / "visbias_neutral12_comparison_summary.csv")
    }
    tests = read_csv(analysis / "all_significance_tests.csv")
    families = Counter(r["correction_family"] for r in tests)
    significant = Counter(
        r["correction_family"] for r in tests if float(r["bh_adjusted_q_value"]) < 0.05
    )
    table = []
    for key in ORDER:
        d = [r for r in dt if r["model_key"] == key]
        b = [r for r in bold if r["model_key"] == key and r.get("toxic_any", "") != ""]
        table.append(
            dict(
                model_key=key,
                model=NAMES[key],
                iat12_gap_pp=100 * float(iat["neutral12"][key]["effect"]),
                iat12_p=float(iat["neutral12"][key]["raw_p_value"]),
                iat12_q=float(iat["neutral12"][key]["bh_adjusted_q_value"]),
                utk_gap_pp=100 * float(iat["utkface"][key]["effect"]),
                utk_p=float(iat["utkface"][key]["raw_p_value"]),
                utk_q=float(iat["utkface"][key]["bh_adjusted_q_value"]),
                dt_agreement_pct=100 * sum(r["classification"] == "AGREE" for r in d) / len(d),
                dt_unclassified_pct=100
                * sum(r["classification"] == "NO_RESPONSE" for r in d)
                / len(d),
                dt_truncated_rows=sum(str(r.get("truncated", "")).lower() == "true" for r in d),
                dt_interpretation="not_interpretable_as_full_response_estimate"
                if key == "gemma_local"
                else "capped_response_classifier_estimate; inspect_generation_quality",
                bold_toxic_pct=100 * sum(int(r["toxic_any"]) for r in b) / len(b),
                visbias12_unsupported_pct=100 * float(vis[key]["unsupported_rate"]),
            )
        )
    write_csv(report / "MAIN_RESULTS_TABLE.csv", table)
    continuity = json.loads((analysis / "gemma_baseline_continuity.json").read_text())
    gemma = table[0]
    directional = [r["model"] for r in table if r["utk_gap_pp"] > 0 and r["dt_agreement_pct"] == 0]
    overview = (
        "This evidence package audits earlier FairFace DPO/LoRA work and completes missing image-based evaluations of pinned local Gemma 3 12B. "
        "Eight hosted models are compared using preserved prompts and responses. New work is inference and analysis only: no base or adapter training, and no replacement of the frozen pre-DPO baseline."
    )
    method = (
        "IAT61 asks for three words from the original 61-word list: 12 faces with 10 repetitions and 400 matched UTKFace images with one repetition. "
        "DecodingTrust uses 48 text stereotype prompts/model; BOLD uses 100 race-focused continuations/model (86 distinct subjects). "
        "VisBias adaptation uses 12 face descriptions and 24 income/education comparisons/model; the original 84-image subset is also retained for eight available models. "
        "Rates and gaps measure prompt-dependent behavior, not an intrinsic unbiased/biased label."
    )
    finding = (
        f"Pinned Gemma's archived 12-face negative-term gap is {gemma['iat12_gap_pp']:.2f} percentage points (pp), "
        f"versus {gemma['utk_gap_pp']:.2f} pp on complete matched UTKFace pairs. Its BOLD toxicity flag rate is {gemma['bold_toxic_pct']:.2f}% "
        f"and neutral12 unsupported image-comparison rate {gemma['visbias12_unsupported_pct']:.2f}%. "
        "Gemma's DT cell is NC (not interpretable): 36/48 responses hit the 150-token cap. Its capped-output counts remain available as diagnostics, not evidence of low bias."
    )
    significance = (
        "Intervals and tests respect the sampling units: image/pair clusters for IAT and VisBias, shared topic blocks for DecodingTrust, and subject clusters for BOLD. "
        "All tests are exploratory with explicit BH correction families; nonsignificance is not evidence of equivalence. "
        "Five of nine UTK tests have q<.05. Unlike the earlier unpaired analysis, these use complete matched-pair sign flips; this post-hoc method change is not new generation for the seven reused models. "
        "Full p/q values and 95% intervals are in all_significance_tests.csv."
    )
    dpo = (
        "FairFace supplied 100 unique images in 50 Black/White matched blocks: 70 grounded-profile preferences and 30 controlled three-word preferences; 80 rows trained, 20 were development-only. "
        "For profiles, chosen text came from gpt-5.6-sol via multimodal Codex CLI, not an OpenAI API snapshot, and rejected text from pinned base Gemma. "
        "Both controlled IAT alternatives came from race-blind templates. Recorded judges were Anthropic claude-sonnet-5 for profiles and claude-opus-5 for IAT, using image-present/text-only and reversed order. "
        "There was no independent human preference review."
    )
    training = (
        "The representative August 10 adapter used modified mean-token DPO (beta 0.05), fresh language-only LoRA rank 8/alpha 16/dropout 0.05, LR 2e-5, BF16, SDPA, "
        "accumulation 8 and 10 optimizer updates. The frozen base remained unchanged. It lowered the 12-face gap from 31.67 to 16.67 pp and Black negativity from 35% to 16.67%, "
        "but retained only 91.11% of response entropy and 81.25% of unique responses. This was the best directional result among five retrospectively inspected configurations, not an accepted successful adapter."
    )
    failures = (
        "Earlier runs either increased Black negativity, reduced the gap by worsening White outcomes, or forced nearly all-positive/repetitive outputs. "
        "The corrected August 10 audit found zero negative-expression IAT blocks in training, although three existed in development. "
        "A diagnostic reached zero negative rates but erased legitimate negative-expression sensitivity. OOM troubleshooting implicated vision-side LoRA and retained paired vision graphs, not eager attention alone."
        " The new local DT run was mostly truncated; the earlier hosted Gemma run also has 34/48 length-limited outputs and is not a valid substitute."
    )
    limits = (
        "The 12-face panel is tiny and repeatedly inspected; the historical FairFace holdout is no longer untouched. File hashes exclude identical images, not repeated identities. "
        "Training shares the IAT prompt family/vocabulary with evaluation. Automated expression, toxicity, sentiment, regard and lexicon scores can be biased or context-insensitive. "
        "Hosted revisions are catalog/provider contracts, not verifiable weight hashes; backend changes and seed nondeterminism remain possible. "
        "No general factual-accuracy test or full post-DPO UTKFace/VisBias/DecodingTrust/BOLD evaluation exists."
    )
    next_steps = (
        "Do not resume a rejected adapter. Freeze a new person-disjoint confirmatory panel; verify expression labels and judge agreement; repair per-split negative-expression coverage; "
        "then conduct a short fresh-LoRA pilot selected only on development data. Require simultaneous gap reduction, grounding/expression preservation, validity and diversity retention, followed by multi-seed replication. "
        "The existing UTKFace future-pool file contains prompts with blank chosen/rejected fields and still needs candidate generation and judging before training."
    )
    lines = [
        "# Research Evidence Report",
        "",
        "September 5, 2026 | Exploratory results | Publication pending approval",
        "",
        "## Scope",
        overview,
        "",
        "## Methods",
        method,
        "",
        "## Benchmark Results",
        "",
        "| Model | IAT12 gap (pp) | UTK matched gap (pp) | DT agree (%) | BOLD flagged (%) | VisBias12 unsupported (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        dt_display = "NC" if r["model_key"] == "gemma_local" else f"{r['dt_agreement_pct']:.2f}"
        lines.append(
            f"| {r['model']} | {r['iat12_gap_pp']:.2f} | {r['utk_gap_pp']:.2f} | {dt_display} | {r['bold_toxic_pct']:.2f} | {r['visbias12_unsupported_pct']:.2f} |"
        )
    lines.extend(["", finding, "", significance, "", "### Significance Families", ""])
    lines.extend(
        f"- {family}: {significant[family]}/{count} tests have BH q <.05."
        for family, count in families.items()
    )
    lines.extend(
        [
            "",
            "### Implicit And Explicit Patterns",
            "Models with a positive descriptive UTKFace gap but zero detected DecodingTrust agreement: "
            + ", ".join(directional)
            + ". This is a descriptive cross-task pattern; it is not proof that they have implicit bias but no explicit bias.",
            "",
            "### Baseline Continuity",
            f"The new local audit matches {continuity['matched_response_tuples']}/{continuity['new_rows']} archived three-word tuples. New Black/White rates were 41.67%/1.67%, versus 35%/3.33% archived. Prompt hash and image ordering agree, but the original full effective runtime state was not recorded; the exact cause is unresolved. The frozen baseline was not replaced. Consult `results/analysis/gemma_baseline_continuity.csv`; model weight hashes were unchanged.",
            "",
            "## FairFace DPO/LoRA",
            dpo,
            "",
            training,
            "",
            "## What Failed",
            failures,
            "",
            "## Limitations",
            limits,
            "",
            "## Next Steps",
            next_steps,
            "",
            "## Evidence And Reproduction",
            f"Absolute package root: `{ROOT}`.",
            "",
            "- [All p-values, adjusted q-values and intervals](../results/analysis/all_significance_tests.csv)",
            "- [Benchmark coverage matrix](benchmark_matrix.csv): complete, historical, not completed and not applicable cells.",
            "- [Researcher-requested variable coverage](requested_variable_coverage.csv) and [column dictionary](DATA_DICTIONARY.csv).",
            "- [Exact DPO spreadsheet](../reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv) and [training record](dpo_training_record.json).",
            "- [Plot registry](plot_registry.csv): every figure, input path, purpose and rebuild command.",
            "- [Methods, metric definitions and academic citations](METHODS.md).",
            "- [Chronological failures and training history](../FINE_TUNING_HISTORY.md).",
            "- [Reproducibility commands and scope limits](../REPRODUCIBILITY.md).",
            "- [Previously published repository, fixed commit](https://github.com/eliaswuberkeley/URAP/tree/7d5c40f2a98b396d230fec8f2f48b6b980dddf76).",
            "",
            "This new package has not been pushed. A future repository link must not be presented as already published. Large model weights and privately held/licensed face images are not proposed GitHub uploads.",
            "",
            "## Figure Guide",
            "",
        ]
    )
    for p in read_csv(report / "plot_registry.csv"):
        lines.extend(
            [f"### {p['title']}", f"[Open figure](../plots/{p['plot_id']}.png)", p["finding"], ""]
        )
    (report / "MAIN_REPORT.md").write_text("\n\n".join(lines) + "\n")
    from reportlab.lib import colors
    from reportlab import rl_config

    rl_config.invariant = 1
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name="Research", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=7)
    )
    styles.add(
        ParagraphStyle(
            name="SmallResearch", fontName="Helvetica", fontSize=7.4, leading=9.5, spaceAfter=5
        )
    )

    def paragraph(text, small=False):
        return Paragraph(escape(text), styles["SmallResearch" if small else "Research"])

    def header(canvas, doc):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#59636E"))
        canvas.drawString(
            40, 24, "URAP | Evidence audit and benchmark completion | September 5, 2026"
        )
        canvas.drawRightString(572, 24, str(doc.page))

    story = [
        Paragraph("Gemma Debiasing and Multimodel Bias Benchmarks", styles["Title"]),
        paragraph(overview),
        Paragraph("Benchmark Design", styles["Heading2"]),
        paragraph(method),
        Paragraph("Results", styles["Heading2"]),
    ]
    headercells = [
        "Model",
        "IAT12 gap\n(pp)",
        "UTK gap\n(pp)",
        "DT agree\n(%)",
        "BOLD flag\n(%)",
        "VisBias12\nunsupported (%)",
    ]
    pdf_table = [headercells] + [
        [
            r["model"].replace(" (local pinned)", " (local)"),
            f"{r['iat12_gap_pp']:.2f}",
            f"{r['utk_gap_pp']:.2f}",
            "NC" if r["model_key"] == "gemma_local" else f"{r['dt_agreement_pct']:.2f}",
            f"{r['bold_toxic_pct']:.2f}",
            f"{r['visbias12_unsupported_pct']:.2f}",
        ]
        for r in table
    ]
    t = Table(pdf_table, colWidths=[147, 65, 65, 60, 65, 80], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EFF5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F6F7")]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            t,
            Spacer(1, 9),
            paragraph(finding),
            paragraph(significance),
            paragraph(
                "Significant tests (BH q<.05): "
                + "; ".join(
                    f"{family.split(':')[0]} {significant[family]}/{count}"
                    for family, count in families.items()
                )
                + ".",
                True,
            ),
            paragraph(
                "Positive IAT gaps mean more negative selections for Black images. DT is overt stereotype agreement; BOLD is automated toxicity >0.3; VisBias counts unsupported A/B choices. These are different constructs and denominators.",
                True,
            ),
            PageBreak(),
            Paragraph("DPO Evidence, Failures and Next Steps", styles["Title"]),
            paragraph(dpo),
            paragraph(training),
            Image(str(ROOT / "plots/10_dpo_tradeoff.png"), width=480, height=240),
            paragraph(
                "Failure audit: no negative-expression IAT blocks reached training; the neutral12 baseline replay also changed despite fixed weights. The frozen comparison is retained, and the replay discrepancy remains unresolved.",
                True,
            ),
            Paragraph("Limitations and Next Steps", styles["Heading2"]),
            paragraph(limits, True),
            paragraph(next_steps, True),
            paragraph(
                "Reproducibility: report/MAIN_REPORT.md links every table/figure; report/DATA_DICTIONARY.csv defines columns; REPRODUCIBILITY.md rebuilds outputs offline. FairFace: Karkkainen & Joo, WACV2021; DPO: Rafailov et al.2023; LoRA: Hu et al.2021; VisBias: Huang et al., EMNLP2025; DecodingTrust: Wang et al., NeurIPS2023; BOLD: Dhamala et al., FAccT2021. Full primary links and caveats: report/METHODS.md. New GitHub publication awaits approval.",
                True,
            ),
        ]
    )
    SimpleDocTemplate(
        str(report / "RESEARCH_REPORT_2PAGE.pdf"),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=38,
    ).build(story, onFirstPage=header, onLaterPages=header)
    from pypdf import PdfReader

    count = len(PdfReader(report / "RESEARCH_REPORT_2PAGE.pdf").pages)
    if count != 2:
        raise ValueError(f"Report must be two pages; generated {count}")
    print("Built two-page PDF, main report and summary table")


if __name__ == "__main__":
    main()
