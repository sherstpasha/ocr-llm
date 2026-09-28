import argparse
import os
import threading
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config_utils import add_strict_output_guard, build_model_output_file, build_openai_chat_completions_url, get_output_folder, load_config_section, resolve_experiment_dir, resolve_path, resolve_stage_model, should_use_openai_chat_api, validate_plain_text_output


DEFAULT_CONFIG_PATH = "configs/config.json"
CONFIG_SECTION = "ocr_llm"

_api_sem = threading.Semaphore(1)


def extract_clean_text(result) -> str:
    if isinstance(result, dict):
        content = None

        if "output" in result:
            output = result["output"]
            if isinstance(output, list) and output:
                preferred_item = None
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message" and item.get("content"):
                        preferred_item = item
                        break
                if preferred_item is None:
                    for item in output:
                        if isinstance(item, dict) and item.get("type") != "reasoning" and item.get("content"):
                            preferred_item = item
                            break
                if preferred_item is None:
                    preferred_item = output[-1]

                if isinstance(preferred_item, dict) and "content" in preferred_item:
                    content = preferred_item["content"]
                else:
                    content = str(preferred_item)
            elif isinstance(output, str):
                content = output

        if not content and "choices" in result and result["choices"]:
            content = result["choices"][0].get("message", {}).get("content")

        if not content and "content" in result:
            content = result["content"]

        if not content:
            content = str(result)

        for prefix in [
            "Here's the corrected text:",
            "Corrected text:",
            "Вот исправленный текст:",
            "Here is the corrected OCR text:",
        ]:
            if content.strip().lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()

        text = content
    else:
        text = str(result)

    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def build_prompt(instruction: str, ocr_text: str) -> str:
    clean_text = " ".join(ocr_text.replace("\n", " ").replace("\r", " ").split())
    return add_strict_output_guard(f"{instruction}\n\nOCR text:\n{clean_text}")


def ask_llm(prompt: str, settings: dict):
    if should_use_openai_chat_api(settings["model"], settings["api_format"]):
        payload = {
            "model": settings["model"],
            "temperature": settings["temperature"],
            "messages": [
                {
                    "role": "system",
                    "content": settings["system_prompt"],
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        if "qwen3.5" in settings["model"].lower():
            payload["enable_thinking"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        request_url = build_openai_chat_completions_url(settings["base_url"])
        headers = {"Content-Type": "application/json"}
        if settings["api_key"] and settings["api_key"] != "dummy":
            headers["Authorization"] = f"Bearer {settings['api_key']}"
    else:
        payload = {
            "model": settings["model"],
            "temperature": settings["temperature"],
            "system_prompt": settings["system_prompt"],
            "input": [
                {
                    "type": "text",
                    "content": prompt,
                }
            ],
        }
        request_url = f"{settings['base_url']}/api/v1/chat"
        headers = {"Content-Type": "application/json"}

    for attempt in range(1, settings["retry_count"] + 1):
        with _api_sem:
            resp = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=600,
            )

        print(f"DEBUG raw response (attempt {attempt}):", resp.status_code, resp.text[:300])
        if resp.status_code == 500 and attempt < settings["retry_count"]:
            print(f"  -> 500 error, retrying in {settings['retry_delay']}s...")
            time.sleep(settings["retry_delay"])
            continue

        resp.raise_for_status()
        return resp.json()

    resp.raise_for_status()


def process_folder(folder: str, settings: dict) -> str:
    folder_path = Path(folder)
    working_folder = get_output_folder(settings["root"], folder_path, settings["experiment_dir"])
    ocr_input_path = working_folder / settings["ocr_input_file"]
    if not ocr_input_path.exists():
        return f"Skipped (no OCR text): {folder_path}"

    working_folder.mkdir(parents=True, exist_ok=True)
    output_path = working_folder / settings["output_file"]
    if settings["skip_done"] and output_path.exists() and output_path.stat().st_size > 0:
        return f"Skipped (already done): {folder_path}"

    with ocr_input_path.open("r", encoding="utf-8") as input_file:
        ocr_text = input_file.read().strip()

    if not ocr_text:
        return f"Skipped (empty OCR text): {folder_path}"

    prompt = build_prompt(settings["instruction"], ocr_text)

    try:
        result = ask_llm(prompt, settings)
        corrected_text = validate_plain_text_output(
            extract_clean_text(result),
            forbid_latin_output=settings["forbid_latin_output"],
        )
    except Exception as exc:
        corrected_text = f"[ERROR processing folder {folder_path}: {exc}]"

    with output_path.open("w", encoding="utf-8") as out:
        out.write(corrected_text)

    print(f"Processed OCR post-correction: {folder_path}")
    return f"Done: {folder_path}"


def run_ocr_llm_with_settings(settings: dict) -> None:
    folders = [
        folder
        for folder, _, files in os.walk(settings["root"])
        if any(file_name.lower().endswith(tuple(settings["image_extensions"])) for file_name in files)
    ]
    folders.sort()

    print(f"Found {len(folders)} folders for OCR post-correction...")
    for folder in tqdm(folders, desc="OCR LLM", unit="document"):
        print(process_folder(folder, settings))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_settings(config_path: str, model_name: str | None = None) -> dict:
    config = load_config_section(config_path, CONFIG_SECTION)
    model = resolve_stage_model(config_path, config, override_model=model_name)
    return {
        "instruction": config["instruction"],
        "model": model,
        "temperature": float(config.get("temperature", 0.0)),
        "system_prompt": config.get("system_prompt", "You are a helpful assistant."),
        "base_url": config.get("base_url", "http://localhost:1234"),
        "api_key": config.get("api_key", "dummy"),
        "root": str(resolve_path(config["root"])),
        "ocr_input_file": config.get("ocr_input_file", "manuscript.txt"),
        "output_file": config.get("output_file") or build_model_output_file("ocr_llm", model),
        "skip_done": config.get("skip_done", True),
        "retry_count": config.get("retry_count", 5),
        "retry_delay": config.get("retry_delay", 10),
        "image_extensions": tuple(config.get("image_extensions", [".png", ".jpg", ".jpeg"])),
        "experiment_dir": resolve_experiment_dir(config_path),
        "api_format": config.get("api_format", "auto"),
        "forbid_latin_output": config.get("forbid_latin_output", True),
    }


def main() -> None:
    args = parse_args()
    run_ocr_llm_with_settings(load_settings(args.config))


if __name__ == "__main__":
    main()
