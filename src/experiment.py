import argparse
import copy
import inspect
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys

from tqdm import tqdm

import requests

from config_utils import build_model_output_file, get_experiment_models, is_section_enabled, load_json_config, model_name_to_suffix, next_experiment_dir, to_project_relative, write_json
from metrics_table import load_settings as load_metrics_settings
from metrics_table import run_metrics_with_settings
from ocr import load_settings as load_ocr_settings
from ocr import run_ocr_with_settings
from ocr_llm import load_settings as load_ocr_llm_settings
from ocr_llm import run_ocr_llm_with_settings
from ocr_vlm import load_settings as load_ocr_vlm_settings
from ocr_vlm import run_ocr_vlm_with_settings
from vlm import load_settings as load_vlm_settings
from vlm import run_vlm_with_settings


DEFAULT_CONFIG_PATH = "configs/config.json"


class Tee:
    """Write stage output to its log and the original terminal stream."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return self.streams[0].encoding


OCR_STAGE = ("ocr", load_ocr_settings, run_ocr_with_settings, None)


MODEL_STAGES = [
    ("vlm", load_vlm_settings, run_vlm_with_settings, "vlm"),
    ("ocr_vlm", load_ocr_vlm_settings, run_ocr_vlm_with_settings, "ocr_vlm"),
    ("ocr_llm", load_ocr_llm_settings, run_ocr_llm_with_settings, "ocr_llm"),
]


def get_enabled_model_stages(config_path: Path) -> list[tuple[str, object, object, str]]:
    return [
        stage
        for stage in MODEL_STAGES
        if is_section_enabled(config_path, stage[0], default=True)
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def get_repeat_count(config_path: str | Path) -> int:
    config_data = load_json_config(config_path)
    experiment_config = config_data.get("experiment", {})
    if not isinstance(experiment_config, dict):
        raise ValueError("'experiment' section must be a JSON object")

    repeat_count = experiment_config.get("repeat_count", 1)
    if not isinstance(repeat_count, int) or repeat_count < 1:
        raise ValueError("'experiment.repeat_count' must be an integer >= 1")

    return repeat_count


def create_experiment_snapshot(
    config_path: str,
    batch_dir: Path,
    run_number: int = 1,
    total_runs: int = 1,
) -> tuple[Path, Path]:
    config_data = load_json_config(config_path)
    experiment_config = config_data.get("experiment", {})
    if not isinstance(experiment_config, dict):
        raise ValueError("'experiment' section must be a JSON object")

    experiment_root = experiment_config.get("root_dir", "experiments")
    experiment_dir = batch_dir / f"repeat_{run_number}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "metrics").mkdir(parents=True, exist_ok=True)

    snapshot = copy.deepcopy(config_data)
    snapshot.setdefault("experiment", {})
    snapshot["experiment"]["root_dir"] = experiment_root
    snapshot["experiment"]["dir"] = to_project_relative(experiment_dir)
    snapshot["experiment"]["batch_dir"] = to_project_relative(batch_dir)
    snapshot["experiment"]["run_number"] = run_number
    snapshot["experiment"]["total_runs"] = total_runs

    snapshot_path = experiment_dir / "config.json"
    write_json(snapshot_path, snapshot)
    return experiment_dir, snapshot_path


def get_base_url(config_path: Path) -> str:
    config_data = load_json_config(config_path)
    common_config = config_data.get("common", {})
    if not isinstance(common_config, dict):
        common_config = {}
    return str(common_config.get("base_url", "http://localhost:1234")).rstrip("/")


def list_loaded_models(base_url: str) -> list[str]:
    models_url = f"{base_url}/models" if base_url.rstrip("/").endswith("/v1") else f"{base_url}/v1/models"
    response = requests.get(models_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", []) if isinstance(data, dict) else []
    return [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]


def load_model(base_url: str, model_name: str) -> dict:
    loaded_models = list_loaded_models(base_url)
    if model_name in loaded_models:
        print(f"[model] '{model_name}' already loaded")
        return {"model": model_name, "status": "already_loaded"}

    response = requests.post(
        f"{base_url}/api/v1/models/load",
        headers={"Content-Type": "application/json"},
        json={"model": model_name},
        timeout=600,
    )
    response.raise_for_status()
    print(f"[model] loaded '{model_name}'")
    return {"model": model_name, "status": "loaded", "response": response.json()}


def unload_model(base_url: str, model_name: str) -> dict:
    response = requests.post(
        f"{base_url}/api/v1/models/unload",
        headers={"Content-Type": "application/json"},
        json={"instance_id": model_name},
        timeout=60,
    )
    response.raise_for_status()
    print(f"[cleanup] unloaded model '{model_name}'")
    return {"model": model_name, "status": "unloaded", "response": response.json()}


def prepare_stage_settings(load_settings_fn, config_path: Path, model_name: str | None, output_prefix: str | None) -> dict:
    signature = inspect.signature(load_settings_fn)
    if "model_name" in signature.parameters:
        settings = load_settings_fn(str(config_path), model_name=model_name)
    else:
        settings = load_settings_fn(str(config_path))

    if model_name is not None and output_prefix is not None:
        settings["model"] = model_name
        settings["output_file"] = build_model_output_file(output_prefix, model_name)
    return settings


def run_stage(stage_name: str, run_fn, settings: dict, log_dir: Path, log_stem: str) -> dict:
    log_path = log_dir / f"{log_stem}.log"
    stage_summary = {
        "name": stage_name,
        "log_file": log_path.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }

    print(f"[{stage_name}] started (log: {log_path})", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = Tee(log_file, sys.__stdout__)
        stderr_tee = Tee(log_file, sys.__stderr__)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            try:
                result = run_fn(settings)
                if isinstance(result, Path):
                    print(f"Result file: {result}")
                stage_summary["status"] = "completed"
            except Exception as exc:
                traceback.print_exc(file=log_file)
                stage_summary["status"] = "failed"
                stage_summary["error"] = str(exc)

    stage_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    print(f"[{stage_name}] {stage_summary['status']}", flush=True)
    return stage_summary


def unload_models(config_path: Path, models: list[str]) -> list[dict]:
    config_data = load_json_config(config_path)
    experiment_config = config_data.get("experiment", {})
    if not isinstance(experiment_config, dict):
        experiment_config = {}

    if experiment_config.get("unload_models_on_finish", True) is False:
        return []

    base_url = get_base_url(config_path)
    results = []

    try:
        loaded_models = set(list_loaded_models(base_url))
    except Exception as exc:
        return [{"status": "failed", "error": f"Could not list loaded models: {exc}"}]

    for model_name in models:
        if model_name not in loaded_models:
            results.append({"model": model_name, "status": "already_unloaded"})
            continue
        try:
            results.append(unload_model(base_url, model_name))
        except Exception as exc:
            results.append({"model": model_name, "status": "failed", "error": str(exc)})
            print(f"[cleanup] failed to unload model '{model_name}': {exc}")

    return results


def get_models_requiring_cleanup(model_runs: list[dict]) -> list[str]:
    cleanup_models = []
    for suite_summary in model_runs:
        model_name = suite_summary.get("model")
        if not isinstance(model_name, str) or not model_name:
            continue

        model_load = suite_summary.get("model_load", {})
        if not isinstance(model_load, dict):
            continue

        if model_load.get("status") not in {"loaded", "already_loaded"}:
            continue

        model_unload = suite_summary.get("model_unload", {})
        if isinstance(model_unload, dict) and model_unload.get("status") == "unloaded":
            continue

        cleanup_models.append(model_name)

    return cleanup_models


def run_model_suite(config_path: Path, log_dir: Path, model_name: str, active_stages: list[tuple[str, object, object, str]]) -> dict:
    suite_key = model_name_to_suffix(model_name)
    suite_summary = {
        "model": model_name,
        "suite_key": suite_key,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "stages": [],
    }

    if not active_stages:
        suite_summary["status"] = "skipped"
        suite_summary["reason"] = "All model stages are disabled in config"
        suite_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return suite_summary

    base_url = get_base_url(config_path)
    try:
        available_models = list_loaded_models(base_url)
        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' is not listed by LM Studio. Available models: {', '.join(available_models) or '(none)'}"
            )
        suite_summary["model_load"] = {
            "model": model_name,
            "status": "available_on_demand",
        }
        print(f"[model] '{model_name}' is available; LM Studio will load it on the first request", flush=True)
    except Exception as exc:
        suite_summary["model_load"] = {"model": model_name, "status": "failed", "error": str(exc)}
        suite_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return suite_summary

    for stage_name, load_settings_fn, run_fn, output_prefix in tqdm(active_stages, desc=f"{model_name} stages", unit="stage"):
        print(f"[model] '{model_name}': starting {stage_name}", flush=True)
        settings = prepare_stage_settings(load_settings_fn, config_path, model_name, output_prefix)
        stage_log_stem = f"{suite_key}_{stage_name}"
        stage_summary = run_stage(f"{stage_name}[{suite_key}]", run_fn, settings, log_dir, stage_log_stem)
        suite_summary["stages"].append(stage_summary)
        if stage_summary["status"] == "failed":
            break

    suite_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return suite_summary


def run_experiment(
    config_path: str,
    batch_dir: Path,
    run_number: int = 1,
    total_runs: int = 1,
) -> tuple[Path, Path]:
    experiment_dir, snapshot_path = create_experiment_snapshot(
        config_path, batch_dir, run_number=run_number, total_runs=total_runs
    )
    log_dir = experiment_dir / "logs"

    summary = {
        "experiment_dir": to_project_relative(experiment_dir),
        "config_file": to_project_relative(snapshot_path),
        "run_number": run_number,
        "total_runs": total_runs,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "ocr_stage": None,
        "model_runs": [],
    }

    experiment_models = get_experiment_models(snapshot_path)
    active_model_stages = get_enabled_model_stages(snapshot_path)
    summary["configured_models"] = experiment_models
    summary["enabled_stages"] = [stage_name for stage_name, *_ in active_model_stages]
    print(f"[experiment] output: {experiment_dir}", flush=True)
    print(f"[experiment] models: {', '.join(experiment_models) or '(none)'}", flush=True)

    ocr_stage_name, ocr_load_settings_fn, ocr_run_fn, _ = OCR_STAGE
    ocr_settings = prepare_stage_settings(ocr_load_settings_fn, snapshot_path, None, None)
    reuse_ocr = bool(load_json_config(config_path).get("experiment", {}).get("reuse_ocr_results", False))
    if reuse_ocr and run_number > 1:
        previous_outputs = batch_dir / f"repeat_{run_number - 1}" / "outputs"
        ocr_settings["reuse_ocr_from"] = previous_outputs
        print(f"[experiment] reusing available OCR results from {previous_outputs}", flush=True)
    ocr_stage = run_stage(ocr_stage_name, ocr_run_fn, ocr_settings, log_dir, ocr_stage_name)
    summary["ocr_stage"] = ocr_stage

    if ocr_stage["status"] != "completed":
        metrics_settings = load_metrics_settings(str(snapshot_path))
        metrics_stage = run_stage("metrics_table", run_metrics_with_settings, metrics_settings, log_dir, "metrics_table")
        summary["metrics_stage"] = metrics_stage
        summary["model_cleanup"] = []
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(experiment_dir / "summary.json", summary)
        print(f"Experiment created: {experiment_dir}")
        print(f"Snapshot config: {snapshot_path}")
        print(f"Summary: {experiment_dir / 'summary.json'}")
        return experiment_dir, snapshot_path

    if active_model_stages:
        for model_name in tqdm(experiment_models, desc="Models", unit="model"):
            suite_summary = run_model_suite(snapshot_path, log_dir, model_name, active_model_stages)
            summary["model_runs"].append(suite_summary)
            if suite_summary.get("model_load", {}).get("status") == "failed":
                break
            if any(stage["status"] != "completed" for stage in suite_summary["stages"]):
                break

    metrics_settings = load_metrics_settings(str(snapshot_path))
    metrics_stage = run_stage("metrics_table", run_metrics_with_settings, metrics_settings, log_dir, "metrics_table")
    summary["metrics_stage"] = metrics_stage

    cleanup_models = get_models_requiring_cleanup(summary["model_runs"])
    summary["model_cleanup"] = unload_models(snapshot_path, cleanup_models) if cleanup_models else []

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(experiment_dir / "summary.json", summary)

    print(f"Experiment created: {experiment_dir}")
    print(f"Snapshot config: {snapshot_path}")
    print(f"Summary: {experiment_dir / 'summary.json'}")

    return experiment_dir, snapshot_path


def main() -> None:
    args = parse_args()
    repeat_count = get_repeat_count(args.config)
    config_data = load_json_config(args.config)
    experiment_config = config_data.get("experiment", {})
    batch_dir = next_experiment_dir(experiment_config.get("root_dir", "experiments"))
    batch_dir.mkdir(parents=True)
    print(f"[experiment] batch output: {batch_dir}", flush=True)

    for run_number in tqdm(range(1, repeat_count + 1), desc="Experiment runs", unit="run"):
        if repeat_count > 1:
            print(f"[batch] starting run {run_number}/{repeat_count}")
        experiment_dir, _ = run_experiment(
            args.config, batch_dir, run_number=run_number, total_runs=repeat_count
        )
        if repeat_count > 1:
            print(f"[batch] completed run {run_number}/{repeat_count} -> {experiment_dir}")


if __name__ == "__main__":
    main()
