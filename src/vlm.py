import argparse
import base64
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from config_utils import add_strict_output_guard, build_model_output_file, build_openai_chat_completions_url, get_output_folder, load_config_section, resolve_experiment_dir, resolve_path, resolve_stage_model, should_use_openai_chat_api, validate_plain_text_output


DEFAULT_CONFIG_PATH = "configs/config.json"
CONFIG_SECTION = "vlm"
MIN_IMAGE_SIZE = 256


def encode_image(image_path: str, max_image_size: int) -> str:
    """Resize image if too large, then encode it as JPEG base64."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_image_size:
            scale = max_image_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def is_retryable_image_error(exc: Exception) -> bool:
    if not isinstance(exc, requests.HTTPError):
        return False

    response = exc.response
    if response is None:
        return False

    body = response.text.lower()
    return "context size has been exceeded" in body or "failed to process image" in body


def get_image_media_type(image_path: str) -> str:
    """Determine media type based on file extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def ask_about_image(
    image_path: str,
    prompt: str,
    max_image_size: int,
    model: str = "google/gemma-3-12b",
    temperature: float = 0.0,
    system_prompt: str = "You are a helpful assistant.",
    base_url: str = "http://localhost:1233",
    api_key: str = "dummy",
    request_timeout: int = 600,
    api_format: str = "auto",
):
    """Send an image + prompt to the API and return the response."""
    current_image_size = max_image_size

    while True:
        base64_image = encode_image(image_path, current_image_size)
        media_type = "image/jpeg"

        try:
            if should_use_openai_chat_api(model, api_format):
                payload = {
                    "model": model,
                    "temperature": temperature,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{base64_image}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        },
                    ],
                }
                if "qwen3.5" in model.lower():
                    payload["enable_thinking"] = False
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

                resp = requests.post(
                    build_openai_chat_completions_url(base_url),
            headers=(
                {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                if api_key and api_key != "dummy"
                else {"Content-Type": "application/json"}
            ),
                    json=payload,
                    timeout=request_timeout,
                )
            else:
                payload = {
                    "model": model,
                    "temperature": temperature,
                    "system_prompt": system_prompt,
                    "input": [
                        {
                            "type": "image",
                            "data_url": f"data:{media_type};base64,{base64_image}",
                        },
                        {
                            "type": "text",
                            "content": prompt,
                        },
                    ],
                }

                resp = requests.post(
                    f"{base_url}/api/v1/chat",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=request_timeout,
                )

            print("DEBUG raw response:", resp.status_code, resp.text[:500])

            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            next_image_size = max(current_image_size // 2, MIN_IMAGE_SIZE)
            if next_image_size < current_image_size and is_retryable_image_error(exc):
                print(f"DEBUG reducing image size from {current_image_size} to {next_image_size} for {Path(image_path).name}")
                current_image_size = next_image_size
                continue
            raise


def get_available_models(base_url: str) -> list[str] | None:
    try:
        models_url = f"{base_url}/models" if base_url.rstrip("/").endswith("/v1") else f"{base_url}/v1/models"
        resp = requests.get(models_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    data = resp.json()
    models = data.get("data", []) if isinstance(data, dict) else []
    return [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]


def ensure_model_available(base_url: str, model: str) -> None:
    available_models = get_available_models(base_url)
    if available_models is None:
        return

    if model not in available_models:
        available_text = ", ".join(available_models) if available_models else "none"
        raise ValueError(
            f"Model '{model}' is not listed by LM Studio. Available models: {available_text}"
        )


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
            "Here's the extracted text from the image, exactly as it appears:",
            "Вот извлечённый текст с изображения, точно как он есть:",
            "Here's the extracted text:",
            "Extracted text:",
        ]:
            if content.strip().lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()

        text = content
    else:
        text = str(result)

    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def load_settings(config_path: str, section_name: str = CONFIG_SECTION, model_name: str | None = None) -> dict:
    config = load_config_section(config_path, section_name)
    model = resolve_stage_model(config_path, config, override_model=model_name)
    return {
        "prompt": add_strict_output_guard(config["prompt"]),
        "model": model,
        "temperature": float(config.get("temperature", 0.0)),
        "system_prompt": config.get("system_prompt", "You are a helpful assistant."),
        "base_url": config.get("base_url", "http://localhost:1234"),
        "api_key": config.get("api_key", "dummy"),
        "root": str(resolve_path(config["root"])),
        "output_file": config.get("output_file") or build_model_output_file("vlm", model),
        "image_extensions": tuple(config.get("image_extensions", [".png", ".jpg", ".jpeg"])),
        "experiment_dir": resolve_experiment_dir(config_path),
        "retry_count": config.get("retry_count", 5),
        "retry_delay": config.get("retry_delay", 10),
        "request_timeout": config.get("request_timeout", 600),
        "max_image_size": config.get("max_image_size", 1536),
        "api_format": config.get("api_format", "auto"),
        "forbid_latin_output": config.get("forbid_latin_output", True),
    }


def run_vlm_with_settings(settings: dict) -> None:
    root = Path(settings["root"])
    image_exts = tuple(settings["image_extensions"])
    model = settings["model"]
    output_file = settings["output_file"]
    experiment_dir = settings["experiment_dir"]

    print(f"Starting VLM run with model '{model}' -> {output_file}")
    ensure_model_available(settings["base_url"], model)

    failed_documents = []

    folders = [(folder, files) for folder, _, files in os.walk(root) if any(name.lower().endswith(image_exts) for name in files)]
    for folder, files in tqdm(folders, desc=f"VLM {model}", unit="document"):
        folder_path = Path(folder)
        image_files = [file_name for file_name in files if file_name.lower().endswith(image_exts)]
        if not image_files:
            continue

        image_files.sort()
        output_folder = get_output_folder(root, folder_path, experiment_dir)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / output_file
        page_texts = []
        folder_errors = []
        for img_name in image_files:
            img_path = folder_path / img_name
            try:
                last_error = None
                one_line_text = None
                for attempt in range(1, settings["retry_count"] + 1):
                    try:
                        result = ask_about_image(
                            image_path=str(img_path),
                            prompt=settings["prompt"],
                            max_image_size=settings["max_image_size"],
                            model=model,
                            temperature=settings["temperature"],
                            system_prompt=settings["system_prompt"],
                            base_url=settings["base_url"],
                            api_key=settings["api_key"],
                            request_timeout=settings["request_timeout"],
                            api_format=settings["api_format"],
                        )
                        one_line_text = validate_plain_text_output(
                            extract_clean_text(result),
                            forbid_latin_output=settings["forbid_latin_output"],
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < settings["retry_count"]:
                            print(f"  -> invalid/failed response for {img_name}, retrying in {settings['retry_delay']}s...")
                            time.sleep(settings["retry_delay"])
                if one_line_text is None:
                    raise last_error or RuntimeError("Unknown VLM processing error")
                page_texts.append(one_line_text)
                print(f"Processed: {img_path}")
            except Exception as exc:
                folder_errors.append(f"{img_name}: {exc}")
                print(f"Failed: {img_path} -> {exc}")

        if folder_errors:
            if output_path.exists():
                output_path.unlink()
            failed_documents.append(f"{folder_path.name} ({'; '.join(folder_errors)})")
            continue

        with output_path.open("w", encoding="utf-8") as out:
            out.write(" ".join(page_texts) + " ")

    if failed_documents:
        raise RuntimeError(
            "VLM stage failed for documents: " + ", ".join(failed_documents)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--section", default=CONFIG_SECTION)
    return parser.parse_args()


def main():
    args = parse_args()
    run_vlm_with_settings(load_settings(args.config, args.section))


if __name__ == "__main__":
    main()
