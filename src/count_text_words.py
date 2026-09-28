from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively count words in every text.txt under a directory.",
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=r"C:\Users\USER\Desktop\metrics\data\handwritten_essay",
        help="Root directory to scan. Defaults to the handwritten_essay test folder.",
    )
    parser.add_argument(
        "--pattern",
        default="text.txt",
        help="File name to search for. Defaults to text.txt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file word counts.",
    )
    return parser.parse_args()


def read_text_file(file_path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp1251")
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def count_words(text: str) -> int:
    return len(text.split())


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_dir}")

    text_files = sorted(path for path in root_dir.rglob(args.pattern) if path.is_file())

    if not text_files:
        print(f"No {args.pattern} files found under: {root_dir}")
        return 0

    total_words = 0
    for file_path in text_files:
        word_count = count_words(read_text_file(file_path))
        total_words += word_count
        if args.verbose:
            print(f"{file_path}: {word_count}")

    print(f"Files found: {len(text_files)}")
    print(f"Total words: {total_words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())