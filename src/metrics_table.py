import json
import os
import re
import time
from pathlib import Path
from typing import List

import argparse

import matplotlib
import pandas as pd
import requests

from config_utils import build_model_output_file, build_openai_chat_completions_url, get_experiment_models, get_metrics_output_dir, get_output_folder, load_config_section, load_json_config, model_name_to_suffix, resolve_experiment_dir, resolve_path, write_json


matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_CONFIG_PATH = "configs/config.json"
CONFIG_SECTION = "metrics_table"

JUDGE_CRITERIA_LABELS = {
    "text_accuracy": "Точность текста",
    "structural_fidelity": "Структурное соответствие",
    "formatting_layout": "Форматирование и макет",
    "completeness": "Полнота",
    "semantic_consistency": "Семантическая согласованность",
    "noise_artifacts": "Шум и артефакты",
}

DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of text recognition quality (OCR / document parsing).

Your task is to compare a reference (ground truth) page text and a predicted page text, and produce a detailed, objective evaluation.

# INPUT

You will receive:

* REFERENCE_TEXT: the correct text of a page
* PREDICTED_TEXT: the recognized / extracted text from the same page

# EVALUATION GOALS

Evaluate how well the predicted text matches the reference at the page level, considering multiple dimensions.

# EVALUATION CRITERIA

1. TEXT ACCURACY

* Character-level correctness
* Word-level correctness
* Missing or hallucinated content

2. STRUCTURAL FIDELITY

* Paragraph boundaries
* Line breaks
* Lists, tables, headings
* Reading order

3. FORMATTING & LAYOUT PRESERVATION

* Titles vs body text
* Sections and hierarchy
* Alignment or grouping (if inferable)

4. COMPLETENESS

* Missing sections
* Truncated text
* Extra unrelated content

5. SEMANTIC CONSISTENCY

* Does the meaning remain intact despite errors?
* Are key entities/numbers preserved correctly?

6. NOISE & ARTIFACTS

* Random characters
* Encoding issues
* OCR artifacts (e.g., \"rn\" vs \"m\")

# SCORING

For each criterion, assign a score from 0 to 5:

0 = completely incorrect
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = near perfect

# OUTPUT FORMAT (STRICT JSON)

Return ONLY valid JSON with the following structure:

{
    \"overall_score\": float,
    \"criteria\": {
        \"text_accuracy\": {\"score\": int, \"justification\": string},
        \"structural_fidelity\": {\"score\": int, \"justification\": string},
        \"formatting_layout\": {\"score\": int, \"justification\": string},
        \"completeness\": {\"score\": int, \"justification\": string},
        \"semantic_consistency\": {\"score\": int, \"justification\": string},
        \"noise_artifacts\": {\"score\": int, \"justification\": string}
    },
    \"major_errors\": [],
    \"minor_errors\": [],
    \"error_examples\": [
        {
            \"reference\": \"...\",
            \"predicted\": \"...\",
            \"issue\": \"...\"
        }
    ],
    \"summary\": \"concise but informative explanation\"
}

# IMPORTANT RULES

* Be strict and evidence-based
* Do NOT assume correctness if unsure
* Do NOT reward partial matches as perfect
* Prefer penalizing hallucinations over omissions
* If structure is lost but text is correct, reflect that in scoring
* Use short, precise justifications (1-2 sentences each)

# EDGE CASES

* If texts are very short, still evaluate all criteria
* If formatting is not inferable, explain and adjust score accordingly
* If predicted text contains content not present in reference -> penalize

# FINAL TASK

Now evaluate the following:

REFERENCE_TEXT:
{{reference_text}}

PREDICTED_TEXT:
{{predicted_text}}"""


def doc_sort_key(path_obj: Path):
    name = path_obj.name
    return (0, int(name)) if name.isdigit() else (1, name)


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    m = len(seq1)
    n = len(seq2)

    if m == 0:
        return n
    if n == 0:
        return m

    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        curr_row = [i] + [0] * n
        a = seq1[i - 1]
        for j in range(1, n + 1):
            b = seq2[j - 1]
            cost = 0 if a == b else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = curr_row

    return prev_row[n]


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def read_text(file_path: Path) -> str:
    return normalize_text(file_path.read_text(encoding="utf-8"))


def count_tokens_gt(text: str) -> int:
    return len(text.split())


def metric_pair(reference_text: str, candidate_path: Path):
    if not candidate_path.exists():
        return pd.NA, pd.NA

    candidate_text = read_text(candidate_path)
    return cer(reference_text, candidate_text), wer(reference_text, candidate_text)


def metric_pair_or_missing(reference_text: str, candidate_path: Path):
    if not candidate_path.exists():
        return "нет", "нет"

    return metric_pair(reference_text, candidate_path)


def round_metric_value(value):
    if pd.isna(value) or isinstance(value, str):
        return value
    return round(float(value), 4)


def normalize_prompt_template(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    return DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE


def method_slug(method_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", method_name.lower())
    return slug.strip("-") or "method"


def extract_response_content(result) -> str:
    if isinstance(result, dict):
        if "output" in result:
            output = result["output"]
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict) and "content" in first:
                    return str(first["content"])
                return str(first)
            if isinstance(output, str):
                return output

        if "choices" in result and result["choices"]:
            return str(result["choices"][0].get("message", {}).get("content", ""))

        if "content" in result:
            return str(result["content"])

    return str(result)


def extract_json_block(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM judge response")

    depth = 0
    in_string = False
    escape = False
    end = -1
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end == -1:
        raise ValueError("No complete JSON object found in LLM judge response")

    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned_candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(cleaned_candidate)


def render_judge_prompt(template: str, reference_text: str, predicted_text: str) -> str:
    return (
        template.replace("{{reference_text}}", reference_text)
        .replace("{{predicted_text}}", predicted_text)
    )


def list_loaded_models(base_url: str) -> list[str]:
    response = requests.get(f"{base_url}/v1/models", timeout=30)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", []) if isinstance(data, dict) else []
    return [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]


def ensure_model_loaded(base_url: str, model_name: str) -> bool:
    loaded_models = list_loaded_models(base_url)
    if model_name in loaded_models:
        return False

    response = requests.post(
        f"{base_url}/api/v1/models/load",
        headers={"Content-Type": "application/json"},
        json={"model": model_name},
        timeout=600,
    )
    response.raise_for_status()
    return True


def unload_model(base_url: str, model_name: str) -> None:
    response = requests.post(
        f"{base_url}/api/v1/models/unload",
        headers={"Content-Type": "application/json"},
        json={"instance_id": model_name},
        timeout=60,
    )
    response.raise_for_status()


def ask_llm_judge(reference_text: str, predicted_text: str, judge_settings: dict) -> dict:
    prompt = render_judge_prompt(judge_settings["prompt_template"], reference_text, predicted_text)
    payload = {
        "model": judge_settings["model"],
        "temperature": judge_settings["temperature"],
        "messages": [
            {"role": "system", "content": judge_settings["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
    }

    last_error = None
    for attempt in range(1, judge_settings["retry_count"] + 1):
        headers = {"Content-Type": "application/json"}
        if judge_settings.get("api_key") and judge_settings["api_key"] != "dummy":
            headers["Authorization"] = f"Bearer {judge_settings['api_key']}"
        response = requests.post(
            build_openai_chat_completions_url(judge_settings["base_url"]),
            headers=headers,
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        content = extract_response_content(response.json())

        try:
            result = extract_json_block(content)
            if "overall_score" not in result:
                raise ValueError("LLM judge response is missing overall_score")
            return result
        except Exception as error:
            last_error = error
            if attempt >= judge_settings["retry_count"]:
                raise ValueError(
                    f"LLM judge returned invalid JSON after {attempt} attempts: {error}"
                ) from error
            time.sleep(judge_settings["retry_delay"])

    raise RuntimeError(f"LLM judge evaluation failed unexpectedly: {last_error}")


def evaluate_llm_judge(reference_text: str, candidate_path: Path, doc_id: str, method_name: str, settings: dict):
    judge_settings = settings["llm_judge"]
    if not judge_settings["enabled"]:
        return pd.NA

    if not candidate_path.exists():
        return "нет"

    details_dir = judge_settings["details_dir"] / str(doc_id)
    details_dir.mkdir(parents=True, exist_ok=True)
    details_path = details_dir / f"{method_slug(method_name)}.json"

    if judge_settings["skip_existing"] and details_path.exists():
        detail = json.loads(details_path.read_text(encoding="utf-8"))
        overall_score = detail.get("overall_score")
        if overall_score is None:
            return pd.NA
        return round(float(overall_score), 4)

    if not judge_settings.get("allow_live_requests", True):
        raise ConnectionError(
            f"LLM judge cache is missing for method '{method_name}' doc '{doc_id}', and LM Studio is unavailable"
        )

    predicted_text = read_text(candidate_path)
    try:
        detail = ask_llm_judge(reference_text, predicted_text, judge_settings)
    except Exception as error:
        write_json(
            details_path,
            {
                "method": method_name,
                "doc_id": str(doc_id),
                "candidate_file": candidate_path.name,
                "error": str(error),
                "overall_score": None,
            },
        )
        return pd.NA

    detail["method"] = method_name
    detail["doc_id"] = str(doc_id)
    detail["candidate_file"] = candidate_path.name
    write_json(details_path, detail)
    return round(float(detail["overall_score"]), 4)


def metric_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    return [column for column in df.columns if column.endswith(f" {suffix}")]


def metric_label(column_name: str, suffix: str) -> str:
    return column_name[: -(len(suffix) + 1)]


def aggregate_metric(series: pd.Series, aggregation: str) -> float:
    numeric_series = pd.to_numeric(series, errors="coerce")
    non_null = numeric_series.dropna()
    if non_null.empty:
        return float("nan")
    if aggregation == "median":
        return float(non_null.median())
    if aggregation == "mean":
        return float(non_null.mean())
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def build_summary_table(df: pd.DataFrame, aggregation: str = "mean") -> pd.DataFrame:
    rows_by_method = {}
    for metric_name in ["CER", "WER", "Judge"]:
        for column in metric_columns(df, metric_name):
            base_label = metric_label(column, metric_name)
            row = rows_by_method.setdefault(base_label, {"Method": base_label})
            row[metric_name] = round(aggregate_metric(df[column], aggregation), 4)

    summary_df = pd.DataFrame(rows_by_method.values())
    if summary_df.empty:
        return summary_df

    if "CER" in summary_df.columns:
        summary_df = summary_df.sort_values("CER").reset_index(drop=True)
    elif "Judge" in summary_df.columns:
        summary_df = summary_df.sort_values("Judge", ascending=False).reset_index(drop=True)
    return summary_df


def build_judge_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for judge_column in metric_columns(df, "Judge"):
        method_name = metric_label(judge_column, "Judge")
        series = pd.to_numeric(df[judge_column], errors="coerce")
        non_null = series.dropna()
        rows.append(
            {
                "Method": method_name,
                "Judge": round(float(non_null.mean()), 4) if not non_null.empty else pd.NA,
                "Judge Std": round(float(non_null.std(ddof=0)), 4) if not non_null.empty else pd.NA,
                "Count": int(non_null.count()),
            }
        )

    judge_df = pd.DataFrame(rows)
    if not judge_df.empty:
        judge_df = judge_df.sort_values("Judge", ascending=False, na_position="last").reset_index(drop=True)
    return judge_df


def criterion_label(criterion_key: str) -> str:
    return JUDGE_CRITERIA_LABELS.get(criterion_key, criterion_key.replace("_", " ").title())


def load_judge_detail_records(details_dir: Path) -> list[dict]:
    if not details_dir.exists():
        return []

    records = []
    for details_path in sorted(details_dir.glob("*/*.json")):
        try:
            detail = json.loads(details_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        method_name = detail.get("method")
        doc_id = detail.get("doc_id")
        criteria = detail.get("criteria")
        if not isinstance(method_name, str) or doc_id is None or not isinstance(criteria, dict):
            continue

        for criterion_key, criterion_payload in criteria.items():
            if not isinstance(criterion_payload, dict):
                continue

            score = criterion_payload.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue

            records.append(
                {
                    "Doc ID": str(doc_id),
                    "Method": method_name,
                    "Criterion Key": criterion_key,
                    "Criterion": criterion_label(criterion_key),
                    "Score": round(score_value, 4),
                }
            )

    return records


def build_judge_criteria_summary_table(judge_details_dir: Path) -> pd.DataFrame:
    records = load_judge_detail_records(judge_details_dir)
    if not records:
        return pd.DataFrame()

    details_df = pd.DataFrame(records)
    summary_df = (
        details_df.groupby(["Criterion", "Method"], as_index=False)
        .agg(
            Score=("Score", "mean"),
            ScoreStd=("Score", lambda values: float(pd.Series(values).std(ddof=0))),
            Count=("Score", "count"),
        )
    )
    summary_df["Score"] = summary_df["Score"].round(4)
    summary_df["ScoreStd"] = summary_df["ScoreStd"].round(4)
    return summary_df.sort_values(["Criterion", "Score"], ascending=[True, False]).reset_index(drop=True)


def build_judge_criteria_per_document_table(judge_details_dir: Path) -> pd.DataFrame:
    records = load_judge_detail_records(judge_details_dir)
    if not records:
        return pd.DataFrame()

    details_df = pd.DataFrame(records)
    details_df["_doc_sort"] = details_df["Doc ID"].apply(
        lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))
    )
    return details_df.sort_values(["Criterion", "_doc_sort", "Method"]).drop(columns=["_doc_sort"]).reset_index(drop=True)


def build_judge_criteria_average_heatmap(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    heatmap_df = summary_df.pivot(index="Method", columns="Criterion", values="Score")
    if heatmap_df.empty:
        return heatmap_df

    method_order = (
        summary_df.groupby("Method", as_index=False)["Score"]
        .mean()
        .sort_values("Score", ascending=False)["Method"]
        .tolist()
    )
    criterion_order = [
        criterion_label(key)
        for key in JUDGE_CRITERIA_LABELS
        if criterion_label(key) in heatmap_df.columns
    ]
    remaining_columns = [column for column in heatmap_df.columns if column not in criterion_order]
    return heatmap_df.reindex(index=method_order, columns=[*criterion_order, *remaining_columns])


def save_criterion_bar_chart(summary_df: pd.DataFrame, criterion_name: str, output_path: Path) -> None:
    criterion_df = summary_df[summary_df["Criterion"] == criterion_name].copy()
    if criterion_df.empty:
        return

    criterion_df = criterion_df.sort_values("Score", ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(4, len(criterion_df) * 0.6)))
    ax.barh(criterion_df["Method"], criterion_df["Score"], color="#1f6f8b")
    ax.set_title(f"Average Judge Criterion: {criterion_name}")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 5)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_criterion_per_document_chart(details_df: pd.DataFrame, criterion_name: str, output_path: Path) -> None:
    criterion_df = details_df[details_df["Criterion"] == criterion_name].copy()
    if criterion_df.empty:
        return

    pivot_df = criterion_df.pivot(index="Doc ID", columns="Method", values="Score")
    if pivot_df.empty:
        return

    pivot_df = pivot_df.reset_index()
    pivot_df["_doc_sort"] = pivot_df["Doc ID"].apply(
        lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))
    )
    pivot_df = pivot_df.sort_values("_doc_sort").drop(columns=["_doc_sort"])

    fig, ax = plt.subplots(figsize=(12, 6))
    doc_ids = pivot_df["Doc ID"].astype(str)
    for column in pivot_df.columns:
        if column == "Doc ID":
            continue
        series = pd.to_numeric(pivot_df[column], errors="coerce")
        ax.plot(doc_ids, series, marker="o", linewidth=2, label=column)

    ax.set_title(f"Judge Criterion by Document: {criterion_name}")
    ax.set_xlabel("Document")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_judge_criteria_artifacts(output_dir: Path, charts_dir: Path, judge_details_dir: Path) -> None:
    summary_df = build_judge_criteria_summary_table(judge_details_dir)
    details_df = build_judge_criteria_per_document_table(judge_details_dir)
    if summary_df.empty or details_df.empty:
        return

    summary_df.to_csv(output_dir / "judge_criteria_summary.csv", index=False, encoding="utf-8-sig")
    details_df.to_csv(output_dir / "judge_criteria_by_document.csv", index=False, encoding="utf-8-sig")
    save_heatmap(
        build_judge_criteria_average_heatmap(summary_df),
        "Средние оценки критериев judge по методам",
        charts_dir / "average_judge_criteria_heatmap.png",
    )

    for criterion_name in summary_df["Criterion"].drop_duplicates().tolist():
        criterion_slug = method_slug(criterion_name)
        save_criterion_bar_chart(summary_df, criterion_name, charts_dir / f"average_judge_{criterion_slug}.png")
        save_criterion_per_document_chart(details_df, criterion_name, charts_dir / f"per_document_judge_{criterion_slug}.png")


def save_bar_chart(summary_df: pd.DataFrame, metric_name: str, output_path: Path, ascending: bool = True) -> None:
    if summary_df.empty or metric_name not in summary_df.columns:
        return

    chart_df = summary_df.sort_values(metric_name, ascending=ascending)
    fig, ax = plt.subplots(figsize=(12, max(4, len(chart_df) * 0.6)))
    ax.barh(chart_df["Method"], chart_df[metric_name], color="#2c7fb8")
    ax.set_title(f"Average {metric_name} by Method")
    ax.set_xlabel(metric_name)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_document_chart(df: pd.DataFrame, suffix: str, output_path: Path) -> None:
    columns = metric_columns(df, suffix)
    if not columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    doc_ids = df["Doc ID"].astype(str)
    for column in columns:
        label = metric_label(column, suffix)
        series = pd.to_numeric(df[column], errors="coerce")
        ax.plot(doc_ids, series, marker="o", linewidth=2, label=label)

    ax.set_title(f"{suffix} by Document")
    ax.set_xlabel("Document")
    ax.set_ylabel(suffix)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_heatmap(dataframe: pd.DataFrame, title: str, output_path: Path) -> None:
    if dataframe.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, dataframe.shape[1] * 1.2), max(4, dataframe.shape[0] * 0.7)))
    values = dataframe.to_numpy(dtype=float)
    heatmap = ax.imshow(values, aspect="auto", cmap="YlGnBu")

    ax.set_title(title)
    ax.set_xticks(range(dataframe.shape[1]))
    ax.set_xticklabels(dataframe.columns, rotation=35, ha="right")
    ax.set_yticks(range(dataframe.shape[0]))
    ax.set_yticklabels(dataframe.index)

    for row_index in range(dataframe.shape[0]):
        for col_index in range(dataframe.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{values[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_average_heatmap(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    metric_names = [column for column in summary_df.columns if column != "Method"]
    return summary_df.set_index("Method")[metric_names]


def build_per_document_heatmap(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    columns = metric_columns(df, suffix)
    if not columns:
        return pd.DataFrame()

    heatmap_df = df[["Doc ID", *columns]].copy()
    heatmap_df[columns] = heatmap_df[columns].apply(pd.to_numeric, errors="coerce")
    renamed_columns = {column: metric_label(column, suffix) for column in columns}
    heatmap_df = heatmap_df.rename(columns=renamed_columns).set_index("Doc ID")
    return heatmap_df


def save_metric_artifacts(df: pd.DataFrame, output_dir: Path, judge_details_dir: Path | None = None) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_table(df)
    median_summary_df = build_summary_table(df, aggregation="median")
    judge_summary_df = build_judge_summary_table(df)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    median_summary_df.to_csv(output_dir / "metrics_median_summary.csv", index=False, encoding="utf-8-sig")
    judge_summary_df.to_csv(output_dir / "judge_summary.csv", index=False, encoding="utf-8-sig")

    save_bar_chart(summary_df, "CER", charts_dir / "average_cer.png")
    save_bar_chart(summary_df, "WER", charts_dir / "average_wer.png")
    save_bar_chart(summary_df, "Judge", charts_dir / "average_judge.png", ascending=False)
    save_per_document_chart(df, "CER", charts_dir / "per_document_cer.png")
    save_per_document_chart(df, "WER", charts_dir / "per_document_wer.png")
    save_per_document_chart(df, "Judge", charts_dir / "per_document_judge.png")
    save_heatmap(build_average_heatmap(summary_df), "Average Metrics Heatmap", charts_dir / "average_metrics_heatmap.png")
    save_heatmap(build_per_document_heatmap(df, "CER"), "CER Heatmap by Document", charts_dir / "per_document_cer_heatmap.png")
    save_heatmap(build_per_document_heatmap(df, "WER"), "WER Heatmap by Document", charts_dir / "per_document_wer_heatmap.png")
    save_heatmap(build_per_document_heatmap(df, "Judge"), "Judge Heatmap by Document", charts_dir / "per_document_judge_heatmap.png")

    if judge_details_dir is not None:
        save_judge_criteria_artifacts(output_dir, charts_dir, judge_details_dir)


def get_stage_file(prefix: str, model_name: str) -> str:
    return build_model_output_file(prefix, model_name)


def build_table(settings: dict) -> pd.DataFrame:
    rows = []
    root = Path(settings["root"])
    stage_enabled = settings["stage_enabled"]
    judge_enabled = settings["llm_judge"]["enabled"]

    for folder in sorted(root.iterdir(), key=doc_sort_key):
        if not folder.is_dir():
            continue

        reference_path = folder / settings["reference_file"]
        if not reference_path.exists():
            continue

        candidate_folder = get_output_folder(root, folder, settings["experiment_dir"])
        reference_text = read_text(reference_path)
        tokens_gt = count_tokens_gt(reference_text)
        ocr_cer, ocr_wer = metric_pair(reference_text, candidate_folder / settings["ocr_file"])
        row = {
            "Doc ID": folder.name,
            "Tokens (GT)": tokens_gt,
            "OCR CER": ocr_cer,
            "OCR WER": ocr_wer,
        }

        if judge_enabled:
            row["OCR Judge"] = evaluate_llm_judge(
                reference_text,
                candidate_folder / settings["ocr_file"],
                folder.name,
                "OCR",
                settings,
            )

        for model_name in settings["models"]:
            model_suffix = model_name_to_suffix(model_name)
            if stage_enabled["vlm"]:
                vlm_path = candidate_folder / get_stage_file("vlm", model_name)
                vlm_cer, vlm_wer = metric_pair_or_missing(reference_text, vlm_path)
                row[f"VLM {model_suffix} CER"] = vlm_cer
                row[f"VLM {model_suffix} WER"] = vlm_wer
                if judge_enabled:
                    row[f"VLM {model_suffix} Judge"] = evaluate_llm_judge(reference_text, vlm_path, folder.name, f"VLM {model_suffix}", settings)

            if stage_enabled["ocr_vlm"]:
                ocr_vlm_path = candidate_folder / get_stage_file("ocr_vlm", model_name)
                ocr_vlm_cer, ocr_vlm_wer = metric_pair_or_missing(reference_text, ocr_vlm_path)
                row[f"OCR_VLM {model_suffix} CER"] = ocr_vlm_cer
                row[f"OCR_VLM {model_suffix} WER"] = ocr_vlm_wer
                if judge_enabled:
                    row[f"OCR_VLM {model_suffix} Judge"] = evaluate_llm_judge(reference_text, ocr_vlm_path, folder.name, f"OCR_VLM {model_suffix}", settings)

            if stage_enabled["ocr_llm"]:
                ocr_llm_path = candidate_folder / get_stage_file("ocr_llm", model_name)
                ocr_llm_cer, ocr_llm_wer = metric_pair_or_missing(reference_text, ocr_llm_path)
                row[f"OCR_LLM {model_suffix} CER"] = ocr_llm_cer
                row[f"OCR_LLM {model_suffix} WER"] = ocr_llm_wer
                if judge_enabled:
                    row[f"OCR_LLM {model_suffix} Judge"] = evaluate_llm_judge(reference_text, ocr_llm_path, folder.name, f"OCR_LLM {model_suffix}", settings)

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["_doc_sort"] = df["Doc ID"].apply(lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))
        df = df.sort_values("_doc_sort").drop(columns=["_doc_sort"]).reset_index(drop=True)

    numeric_cols = ["OCR CER", "OCR WER"]
    text_or_numeric_cols = []
    if judge_enabled:
        numeric_cols.append("OCR Judge")

    for model_name in settings["models"]:
        model_suffix = model_name_to_suffix(model_name)
        if stage_enabled["vlm"]:
            numeric_cols.extend([
                f"VLM {model_suffix} CER",
                f"VLM {model_suffix} WER",
            ])
            if judge_enabled:
                numeric_cols.append(f"VLM {model_suffix} Judge")

        if stage_enabled["ocr_vlm"]:
            text_or_numeric_cols.extend([
                f"OCR_VLM {model_suffix} CER",
                f"OCR_VLM {model_suffix} WER",
            ])
            if judge_enabled:
                text_or_numeric_cols.append(f"OCR_VLM {model_suffix} Judge")

        if stage_enabled["ocr_llm"]:
            text_or_numeric_cols.extend([
                f"OCR_LLM {model_suffix} CER",
                f"OCR_LLM {model_suffix} WER",
            ])
            if judge_enabled:
                text_or_numeric_cols.append(f"OCR_LLM {model_suffix} Judge")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    for col in text_or_numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(round_metric_value)

    return df


def load_settings(config_path: str) -> dict:
    data = load_json_config(config_path)
    config = load_config_section(config_path, CONFIG_SECTION)
    root = resolve_path(config["root"])
    experiment_dir = resolve_experiment_dir(config_path)
    experiment_config = data.get("experiment", {})
    if not isinstance(experiment_config, dict):
        experiment_config = {}

    ocr_section = data.get("ocr", {})
    common_config = data.get("common", {})
    if not isinstance(common_config, dict):
        common_config = {}
    models = get_experiment_models(config_path)

    llm_judge_config = data.get("llm_judge", {})
    if not isinstance(llm_judge_config, dict):
        llm_judge_config = {}

    return {
        "root": root,
        "experiment_dir": experiment_dir,
        "reference_file": config.get("reference_file", "text.txt"),
        "ocr_file": config.get("ocr_file", ocr_section.get("output_file", "ocr.txt")),
        "models": models,
        "stage_enabled": {
            "vlm": bool(data.get("vlm", {}).get("enabled", True)) if isinstance(data.get("vlm", {}), dict) else True,
            "ocr_vlm": bool(data.get("ocr_vlm", {}).get("enabled", True)) if isinstance(data.get("ocr_vlm", {}), dict) else True,
            "ocr_llm": bool(data.get("ocr_llm", {}).get("enabled", True)) if isinstance(data.get("ocr_llm", {}), dict) else True,
        },
        "output_csv": config.get("output_csv", "metrics_table.csv"),
        "output_dir": get_metrics_output_dir(experiment_dir, config.get("output_dir", "output")),
        "llm_judge": {
            "enabled": llm_judge_config.get("enabled", False),
            "model": llm_judge_config.get("model", "gemma-3-12b-it"),
            "temperature": float(llm_judge_config.get("temperature", 0.0)),
            "system_prompt": llm_judge_config.get("system_prompt", "Return valid JSON only."),
            "prompt_template": normalize_prompt_template(llm_judge_config.get("prompt_template", DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE)),
            "base_url": str(llm_judge_config.get("base_url", common_config.get("base_url", "http://localhost:1234"))).rstrip("/"),
            "api_key": os.environ.get("LLM_API_KEY", common_config.get("api_key", "dummy")),
            "retry_count": llm_judge_config.get("retry_count", 3),
            "retry_delay": llm_judge_config.get("retry_delay", 5),
            "skip_existing": llm_judge_config.get("skip_existing", True),
            "details_subdir": llm_judge_config.get("details_subdir", "llm_judge"),
            "unload_on_finish": llm_judge_config.get("unload_on_finish", True),
        },
    }


def run_metrics_with_settings(settings: dict) -> Path:
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    settings["llm_judge"]["details_dir"] = output_dir / settings["llm_judge"]["details_subdir"]

    settings["llm_judge"]["allow_live_requests"] = settings["llm_judge"]["enabled"]
    df = build_table(settings)

    output_path = output_dir / settings["output_csv"]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_metric_artifacts(df, output_dir, settings["llm_judge"]["details_dir"])

    print(f"Saved: {output_path}")
    print(df.to_string(index=False))
    return output_path


def main() -> None:
    args = parse_args()
    run_metrics_with_settings(load_settings(args.config))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    main()
