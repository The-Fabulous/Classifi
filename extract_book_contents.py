import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pymupdf


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "book_import_config.json"
CHAPTER_NUMBER_PATTERN = re.compile(r"^\d{1,2}$")
SUBTOPIC_NUMBER_PATTERN = re.compile(r"^\d{1,2}\.\d{1,2}\.?$")
APPENDIX_SUBTOPIC_PATTERN = re.compile(r"^[A-Z]\.\d{1,2}\.?$")
INLINE_CHAPTER_PATTERN = re.compile(r"^(\d{1,2})\s+(.+)$")
INLINE_SUBTOPIC_PATTERN = re.compile(r"^(\d{1,2}\.\d{1,2}\.?|[A-Z]\.\d{1,2}\.?)\s+(.+)$")
PAGE_NUMBER_SUFFIX_PATTERN = re.compile(r"\s+(?:\d+|[ivxlcdm]+)$", re.IGNORECASE)


def read_export_config(config_path: Path) -> tuple[Path, Path, int, int]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read config file {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("The config file must contain a JSON object.")

    pdf_value = config.get("pdf_path")
    if not isinstance(pdf_value, str) or not pdf_value.strip():
        raise ValueError("pdf_path must be a non-empty path.")
    start_page = config.get("start_page")
    end_page = config.get("end_page")
    if type(start_page) is not int or type(end_page) is not int:
        raise ValueError("start_page and end_page must be integers.")
    if start_page < 1 or end_page < start_page:
        raise ValueError("Page numbers must be positive and end_page >= start_page.")
    output_filename = config.get("contents_output_filename", "book_contents.json")
    if (
        not isinstance(output_filename, str)
        or not output_filename.strip()
        or Path(output_filename).name != output_filename
        or not output_filename.endswith(".json")
    ):
        raise ValueError("contents_output_filename must be a .json filename, not a path.")

    pdf_path = Path(pdf_value)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_DIR / pdf_path
    output_path = PROJECT_DIR / "materials" / output_filename
    return pdf_path, output_path, start_page, end_page


def parse_contents_pages(page_texts: list[str]) -> dict[str, list[dict[str, Any]]]:
    topics: list[dict[str, Any]] = []
    current_topic: dict[str, Any] | None = None

    lines = [line.strip() for text in page_texts for line in text.splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index]
        inline_chapter = INLINE_CHAPTER_PATTERN.fullmatch(line)
        inline_subtopic = INLINE_SUBTOPIC_PATTERN.fullmatch(line)
        if inline_chapter:
            title = PAGE_NUMBER_SUFFIX_PATTERN.sub("", inline_chapter.group(2)).strip()
            current_topic = {
                "topic": f"{inline_chapter.group(1)}. {title}",
                "subtopics": [],
            }
            topics.append(current_topic)
        elif inline_subtopic and current_topic is not None:
            title = PAGE_NUMBER_SUFFIX_PATTERN.sub("", inline_subtopic.group(2)).strip()
            number = inline_subtopic.group(1).rstrip(".")
            current_topic["subtopics"].append(f"{number}. {title}")
        elif CHAPTER_NUMBER_PATTERN.fullmatch(line):
            title_index = index + 1
            while title_index < len(lines) and not lines[title_index]:
                title_index += 1
            if title_index < len(lines):
                title = PAGE_NUMBER_SUFFIX_PATTERN.sub("", lines[title_index]).strip()
                if title:
                    current_topic = {
                        "topic": f"{line}. {title}",
                        "subtopics": [],
                    }
                    topics.append(current_topic)
                    index = title_index
        elif (
            SUBTOPIC_NUMBER_PATTERN.fullmatch(line)
            or APPENDIX_SUBTOPIC_PATTERN.fullmatch(line)
        ) and current_topic is not None:
            title_index = index + 1
            while title_index < len(lines) and not lines[title_index]:
                title_index += 1
            if title_index < len(lines):
                title = PAGE_NUMBER_SUFFIX_PATTERN.sub("", lines[title_index]).strip()
                if title:
                    number = line.rstrip(".")
                    current_topic["subtopics"].append(f"{number}. {title}")
                    index = title_index
        index += 1

    if not topics:
        raise ValueError("No chapter entries found in the selected PDF pages.")
    return {"topics": topics}


def create_book_contents(
    pdf_path: Path, start_page: int, end_page: int
) -> dict[str, list[dict[str, Any]]]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        with pymupdf.open(pdf_path) as document:
            if end_page > document.page_count:
                raise ValueError(
                    f"end_page is {end_page}, but the PDF has only "
                    f"{document.page_count} pages."
                )
            page_texts = [
                document[page_number - 1].get_text()
                for page_number in range(start_page, end_page + 1)
            ]
            return parse_contents_pages(page_texts)
    except pymupdf.FileDataError as error:
        raise ValueError(f"Could not open PDF {pdf_path}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a JSON topic list using book_import_config.json."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Import config to read (default: book_import_config.json)",
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_DIR / config_path
        pdf_path, output_path, start_page, end_page = read_export_config(config_path)
        contents = create_book_contents(pdf_path, start_page, end_page)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(contents, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        print(f"Book contents export failed: {error}", file=sys.stderr)
        return 1

    topic_count = len(contents["topics"])
    subtopic_count = sum(len(topic["subtopics"]) for topic in contents["topics"])
    print(f"Created {output_path} ({topic_count} topics, {subtopic_count} subtopics).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
