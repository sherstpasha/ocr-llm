import argparse
import base64
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from config_utils import add_strict_output_guard, build_model_output_file, build_openai_chat_completions_url, get_output_folder, load_config_section, resolve_experiment_dir, resolve_path, resolve_stage_model, should_use_openai_chat_api, validate_plain_text_output


DEFAULT_CONFIG_PATH = "configs/config.json"
CONFIG_SECTION = "ocr_vlm"
MIN_IMAGE_SIZE = 256

# LM Studio обычно держит только 1 запрос за раз — семафор не даёт его завалить
_api_sem = threading.Semaphore(1)


def encode_image(image_path: str, max_image_size: int) -> str:
    """Resize image if too large, then encode to base64."""
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
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def build_hybrid_prompt(base_prompt: str, ocr_hint_text: str) -> str:
    clean_hint = " ".join(ocr_hint_text.replace("\n", " ").replace("\r", " ").split())
    return add_strict_output_guard(
        (
        f"{base_prompt}\n\n"
        "OCR hint text:\n"
        f"{clean_hint}"
        )
    )


def ask_about_images(
    image_paths,
    prompt: str,
    max_image_size: int,
    model: str = "google/gemma-3-12b",
    temperature: float = 0.0,
    system_prompt: str = "You are a helpful assistant.",
    base_url: str = "http://localhost:1233",
    api_key: str = "dummy",
    retry_count: int = 5,
    retry_delay: int = 10,
    api_format: str = "auto",
):
    current_image_size = max_image_size

    while True:
        if should_use_openai_chat_api(model, api_format):
            user_content = []
            for image_path in image_paths:
                base64_image = encode_image(image_path, current_image_size)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    }
                )

            user_content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

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
                        "content": user_content,
                    },
                ],
            }
            if "qwen3.5" in model.lower():
                payload["enable_thinking"] = False
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            request_url = build_openai_chat_completions_url(base_url)
            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "dummy":
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            payload_input = []
            for image_path in image_paths:
                media_type = get_image_media_type(image_path)
                base64_image = encode_image(image_path, current_image_size)
                payload_input.append(
                    {
                        "type": "image",
                        "data_url": f"data:{media_type};base64,{base64_image}",
                    }
                )

            payload_input.append(
                {
                    "type": "text",
                    "content": prompt,
                }
            )

            payload = {
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "input": payload_input,
            }
            request_url = f"{base_url}/api/v1/chat"
            headers = {"Content-Type": "application/json"}

        last_exc = None
        for attempt in range(1, retry_count + 1):
            try:
                with _api_sem:
                    resp = requests.post(
                        request_url,
                        headers=headers,
                        json=payload,
                        timeout=600,
                    )
                print(f"DEBUG raw response (attempt {attempt}):", resp.status_code, resp.text[:300])
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < retry_count:
                    print(f"  -> request failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

        next_image_size = max(current_image_size // 2, MIN_IMAGE_SIZE)
        if next_image_size < current_image_size and last_exc is not None and is_retryable_image_error(last_exc):
            print(f"DEBUG reducing batch image size from {current_image_size} to {next_image_size}")
            current_image_size = next_image_size
            continue

        raise last_exc or RuntimeError("Unknown OCR+VLM request error")


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


def chunk_items(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return [items]
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def process_folder(folder: str, settings: dict) -> str:
    """Process a single folder: read OCR hint, query VLM once for all images, write output."""
    image_exts = (".png", ".jpg", ".jpeg")
    folder_path = Path(folder)
    source_root = Path(settings["root"])
    files = os.listdir(folder_path)
    image_files = sorted(f for f in files if f.lower().endswith(image_exts))

    if not image_files:
        return f"Skipped (no images): {folder_path}"

    working_folder = get_output_folder(source_root, folder_path, settings["experiment_dir"])
    ocr_hint_path = working_folder / settings["ocr_hint_file"]
    if not ocr_hint_path.exists():
        return f"Skipped (no OCR hint): {folder_path}"

    working_folder.mkdir(parents=True, exist_ok=True)
    output_path = working_folder / settings["output_file"]
    if settings["skip_done"] and output_path.exists() and output_path.stat().st_size > 0:
        return f"Skipped (already done): {folder_path}"

    with ocr_hint_path.open("r", encoding="utf-8") as hint_file:
        ocr_hint_text = hint_file.read()

    hybrid_prompt = build_hybrid_prompt(settings["base_prompt"], ocr_hint_text)

    image_paths = [str(folder_path / img_name) for img_name in image_files]
    image_batches = chunk_items(image_paths, settings["images_per_request"])

    try:
        batch_results = []
        for batch_paths in image_batches:
            result = ask_about_images(
                image_paths=batch_paths,
                prompt=hybrid_prompt,
                max_image_size=settings["max_image_size"],
                model=settings["model"],
                temperature=settings["temperature"],
                system_prompt=settings["system_prompt"],
                base_url=settings["base_url"],
                api_key=settings["api_key"],
                retry_count=settings["retry_count"],
                retry_delay=settings["retry_delay"],
                api_format=settings["api_format"],
            )
            batch_results.append(
                validate_plain_text_output(
                    extract_clean_text(result),
                    forbid_latin_output=settings["forbid_latin_output"],
                )
            )
        one_line_text = " ".join(batch_results)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"{folder_path.name}: {exc}") from exc

    with output_path.open("w", encoding="utf-8") as out:
        out.write(one_line_text)

    print(f"Processed {len(image_paths)} images: {folder_path}")

    return f"Done: {folder_path}"


def run_ocr_vlm_with_settings(settings: dict) -> None:
    image_exts = tuple(settings["image_extensions"])

    # Собираем все папки с изображениями
    folders = [
        folder
        for folder, _, files in os.walk(settings["root"])
        if any(f.lower().endswith(image_exts) for f in files)
    ]
    folders.sort()

    print(f"Found {len(folders)} folders, running with {settings['workers']} workers...")

    failed_documents = []
    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        futures = {executor.submit(process_folder, f, settings): f for f in folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="OCR+VLM", unit="document"):
            try:
                print(future.result())
            except Exception as exc:
                failed_documents.append(str(exc))
                print(f"Failed: {exc}")

    if failed_documents:
        raise RuntimeError(
            "OCR_VLM stage failed for documents: " + ", ".join(failed_documents)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_settings(config_path: str, model_name: str | None = None) -> dict:
    config = load_config_section(config_path, CONFIG_SECTION)
    model = resolve_stage_model(config_path, config, override_model=model_name)
    return {
        "base_prompt": config["base_prompt"],
        "model": model,
        "temperature": float(config.get("temperature", 0.0)),
        "system_prompt": config.get("system_prompt", "You are a helpful assistant."),
        "base_url": config.get("base_url", "http://localhost:1234"),
        "api_key": config.get("api_key", "dummy"),
        "root": str(resolve_path(config["root"])),
        "ocr_hint_file": config.get("ocr_hint_file", "manuscript.txt"),
        "output_file": config.get("output_file") or build_model_output_file("ocr_vlm", model),
        "workers": config.get("workers", 2),
        "skip_done": config.get("skip_done", True),
        "retry_count": config.get("retry_count", 5),
        "retry_delay": config.get("retry_delay", 10),
        "max_image_size": config.get("max_image_size", 1536),
        "images_per_request": config.get("images_per_request", 1),
        "image_extensions": config.get("image_extensions", [".png", ".jpg", ".jpeg"]),
        "experiment_dir": resolve_experiment_dir(config_path),
        "api_format": config.get("api_format", "auto"),
        "forbid_latin_output": config.get("forbid_latin_output", True),
    }


def main():
    args = parse_args()
    run_ocr_vlm_with_settings(load_settings(args.config))


if __name__ == "__main__":
    main()
