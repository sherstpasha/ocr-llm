import argparse
import os
import shutil
from pathlib import Path

from manuscript import Pipeline
from tqdm import tqdm

from config_utils import copy_source_text_file, get_output_folder, load_config_section, resolve_experiment_dir, resolve_path


DEFAULT_CONFIG_PATH = "configs/config.json"
CONFIG_SECTION = "ocr"


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
	return parser.parse_args()


def load_settings(config_path: str) -> dict:
	config = load_config_section(config_path, CONFIG_SECTION)
	return {
		"root": resolve_path(config["root"]),
		"output_file": config.get("output_file", "ocr.txt"),
		"image_extensions": tuple(config.get("image_extensions", [".png", ".jpg", ".jpeg"])),
		"experiment_dir": resolve_experiment_dir(config_path),
	}


def run_ocr_with_settings(settings: dict) -> None:
	root = Path(settings["root"])
	image_exts = tuple(settings["image_extensions"])
	experiment_dir = settings["experiment_dir"]

	pipeline = None

	folders = []
	for folder, _, files in os.walk(root):
		if any(f.lower().endswith(image_exts) for f in files):
			folders.append((folder, files))

	with tqdm(total=len(folders), desc="OCR", unit="doc", dynamic_ncols=True) as progress:
		for folder, files in folders:
			folder_path = Path(folder)
			progress.set_postfix_str(str(folder_path), refresh=True)
			image_files = [f for f in files if f.lower().endswith(image_exts)]
			image_files.sort()
			output_folder = get_output_folder(root, folder_path, experiment_dir)
			output_folder.mkdir(parents=True, exist_ok=True)
			copy_source_text_file(folder_path, output_folder)
			output_file = output_folder / settings["output_file"]
			reuse_root = settings.get("reuse_ocr_from")
			if reuse_root:
				relative_folder = folder_path.relative_to(root)
				reused_file = Path(reuse_root) / relative_folder / settings["output_file"]
				if reused_file.is_file() and reused_file.stat().st_size > 0:
					shutil.copyfile(reused_file, output_file)
					progress.update(1)
					continue
			if pipeline is None:
				pipeline = Pipeline()
			with output_file.open("w", encoding="utf-8") as out:
				for img_name in image_files:
					img_path = folder_path / img_name
					try:
						result = pipeline.predict(str(img_path))
						text = pipeline.get_text(result["page"])
					except Exception as exc:
						text = f"[ERROR processing {img_name}: {exc}]"

					one_line_text = " ".join(text.strip().splitlines())
					out.write(one_line_text + " ")
			progress.update(1)


def main() -> None:
	args = parse_args()
	run_ocr_with_settings(load_settings(args.config))


if __name__ == "__main__":
	main()
