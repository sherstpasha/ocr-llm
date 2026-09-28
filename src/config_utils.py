import json
import os
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REASONING_LEAK_MARKERS = [
    "thinking process",
    "the user wants",
    "analyze the image",
    "transcribe line by line",
    "step-by-step correction",
    "drafting the correction",
    "line 1:",
]
LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
        "a": "а",
        "c": "с",
        "e": "е",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
    }
)
VISION_MODEL_MARKERS = (
    "gemma-3",
    "internvl",
    "llava",
    "minicpm",
    "molmo",
    "phi-3.5-vision",
    "pixtral",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen2.5vl",
    "vision",
    "vlm",
)


def load_json_config(config_path: str | Path) -> dict:
    load_dotenv()
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")

    return data


def load_dotenv() -> None:
    """Load project .env values without overriding variables already in the shell."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


def load_config_section(config_path: str | Path, section_name: str) -> dict:
    data = load_json_config(config_path)

    if section_name not in data:
        return data

    common = data.get("common", {})
    section = data[section_name]

    if not isinstance(common, dict):
        raise ValueError(f"Common config must be a JSON object in {config_path}")
    if not isinstance(section, dict):
        raise ValueError(f"Section '{section_name}' must be a JSON object in {config_path}")

    merged = {**common, **section}
    merged["api_key"] = os.environ.get("LLM_API_KEY", merged.get("api_key", "dummy"))
    return merged


def is_section_enabled(config_path: str | Path, section_name: str, default: bool = True) -> bool:
    data = load_json_config(config_path)
    section = data.get(section_name, {})
    if not isinstance(section, dict):
        return default

    enabled = section.get("enabled", default)
    if isinstance(enabled, bool):
        return enabled
    return default


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_experiment_config(config_path: str | Path) -> dict:
    data = load_json_config(config_path)
    experiment = data.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError(f"Experiment config must be a JSON object in {config_path}")
    return experiment


def get_experiment_models(config_path: str | Path) -> list[str]:
    experiment = load_experiment_config(config_path)
    models = experiment.get("models")
    if isinstance(models, list) and all(isinstance(item, str) and item.strip() for item in models):
        return [item.strip() for item in models]

    data = load_json_config(config_path)
    discovered = []
    for section_name in ["vlm", "ocr_vlm", "ocr_llm"]:
        section = data.get(section_name, {})
        if not isinstance(section, dict):
            continue

        model_name = section.get("model")
        if isinstance(model_name, str) and model_name.strip() and model_name.strip() not in discovered:
            discovered.append(model_name.strip())

    return discovered


def resolve_stage_model(config_path: str | Path, config: dict, override_model: str | None = None) -> str:
    if override_model is not None:
        model_name = override_model.strip()
        if model_name:
            return model_name

    configured_model = config.get("model")
    if isinstance(configured_model, str) and configured_model.strip():
        return configured_model.strip()

    experiment_models = get_experiment_models(config_path)
    if len(experiment_models) == 1:
        return experiment_models[0]

    if len(experiment_models) > 1:
        raise ValueError(
            "Model is not specified for this stage. Use src/experiment.py for multi-model runs or provide a stage-specific model."
        )

    raise ValueError(
        "Model is not specified. Set section 'model' or provide experiment.models with a single model."
    )


def resolve_experiment_dir(config_path: str | Path) -> Path | None:
    experiment = load_experiment_config(config_path)
    experiment_dir = experiment.get("dir")
    if not experiment_dir:
        return None
    return resolve_path(experiment_dir)


def get_output_folder(root: str | Path, source_folder: str | Path, experiment_dir: str | Path | None) -> Path:
    source_root = Path(root)
    folder_path = Path(source_folder)
    if experiment_dir is None:
        return folder_path
    relative_folder = folder_path.relative_to(source_root)
    return Path(experiment_dir) / "outputs" / relative_folder


def get_metrics_output_dir(experiment_dir: str | Path | None, fallback_output_dir: str | Path) -> Path:
    if experiment_dir is not None:
        return Path(experiment_dir) / "metrics"
    return resolve_path(fallback_output_dir)


def copy_source_text_file(source_folder: str | Path, output_folder: str | Path, file_name: str = "text.txt") -> Path | None:
    source_path = Path(source_folder) / file_name
    if not source_path.exists() or not source_path.is_file():
        return None

    target_path = Path(output_folder) / file_name
    if source_path.resolve() == target_path.resolve():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    return target_path


def next_experiment_dir(root_dir: str | Path) -> Path:
    base_dir = resolve_path(root_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = base_dir / "exp"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = base_dir / f"exp{index}"
        if not candidate.exists():
            return candidate
        index += 1


def to_project_relative(path_value: str | Path) -> str:
    path = Path(path_value)
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path
    return relative.as_posix()


def write_json(path: str | Path, data: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def model_name_to_suffix(model_name: str) -> str:
    tail = model_name.rsplit("/", 1)[-1].strip().lower()
    suffix = re.sub(r"[^a-z0-9._-]+", "-", tail)
    suffix = suffix.strip("-._")
    if not suffix:
        raise ValueError(f"Cannot build file suffix from model name: {model_name}")
    return suffix


def build_model_output_file(prefix: str, model_name: str, extension: str = ".txt") -> str:
    suffix = model_name_to_suffix(model_name)
    clean_extension = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}_{suffix}{clean_extension}"


def add_strict_output_guard(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "IMPORTANT OUTPUT RULES:\n"
        "- Think silently. Do not reveal reasoning.\n"
        "- Return only the final recognized/corrected text.\n"
        "- Do not write analysis, headings, notes, or explanations.\n"
        "- Do not use English words if the source text is Russian.\n"
        "- Do not output phrases like 'The user wants', 'Thinking Process', 'Analyze the image', or similar."
    )


def should_use_openai_chat_api(model_name: str, api_format: str = "auto") -> bool:
    normalized = (api_format or "auto").strip().lower()
    if normalized == "openai":
        return True
    if normalized == "legacy":
        return False

    lowered_model = model_name.strip().lower()
    return "qwen3.5" in lowered_model


def is_vision_model(model_name: str) -> bool:
    lowered_model = model_name.strip().lower()
    return any(marker in lowered_model for marker in VISION_MODEL_MARKERS)


def build_openai_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def normalize_mixed_script_text(text: str) -> str:
    parts = []
    for token in text.split():
        if re.search(r"[А-Яа-яЁё]", token):
            parts.append(token.translate(LATIN_TO_CYRILLIC))
        else:
            parts.append(token)
    return " ".join(parts)


def strip_reasoning_preamble(text: str) -> str:
    cleaned = text.replace("\r", "")
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*</think>\s*", "", cleaned, flags=re.IGNORECASE)

    if re.search(r"[A-Za-z]", cleaned) and re.search(r"[А-Яа-яЁё]", cleaned):
        first_cyrillic = re.search(r"[А-Яа-яЁё]", cleaned)
        if first_cyrillic and first_cyrillic.start() > 40:
            latin_prefix = cleaned[: first_cyrillic.start()]
            if len(re.findall(r"[A-Za-z]{2,}", latin_prefix)) >= 3:
                cleaned = cleaned[first_cyrillic.start() :]

    cleaned = re.sub(r"^\s*(?:[-*\d.)\s]+)?(?:thinking process|analy(?:s|z)e the request|based on the visual content).*?(?=[А-Яа-яЁё])", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def has_excessive_latin_text(text: str) -> bool:
    latin_words = re.findall(r"[A-Za-z]{3,}", text)
    if not latin_words:
        return False

    latin_chars = sum(len(word) for word in latin_words)
    cyrillic_chars = len(re.findall(r"[А-Яа-яЁё]", text))

    if len(latin_words) >= 3:
        return True

    if latin_chars >= 12 and latin_chars * 6 >= max(cyrillic_chars, 1):
        return True

    return False


def validate_plain_text_output(text: str, forbid_latin_output: bool = True) -> str:
    cleaned = normalize_mixed_script_text(
        " ".join(strip_reasoning_preamble(text).replace("\n", " ").replace("\r", " ").split())
    )
    lowered = cleaned.lower()

    for marker in REASONING_LEAK_MARKERS:
        if marker in lowered:
            raise ValueError(f"Model leaked reasoning marker: {marker}")

    if forbid_latin_output and has_excessive_latin_text(cleaned):
        raise ValueError("Model returned Latin text in a Russian-only task")

    return cleaned
