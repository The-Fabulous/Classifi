import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import import_book
from tutor_graph import LLMRequestError


class BookImportTests(unittest.TestCase):
    def test_reads_valid_config(self):
        config = {
            "pdf_path": "books/book.pdf",
            "start_page": 2,
            "end_page": 4,
            "chunk_size": 2,
            "article_id": "chapter-one",
            "title": "Chapter One",
            "overview": "A short overview.",
            "output_filename": "chapter-one.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(import_book.read_config(config_path), config)

    def test_rejects_invalid_page_range(self):
        config = {
            "pdf_path": "books/book.pdf",
            "start_page": 5,
            "end_page": 4,
            "chunk_size": 2,
            "article_id": "chapter-one",
            "title": "Chapter One",
            "overview": "A short overview.",
            "output_filename": "chapter-one.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "end_page"):
                import_book.read_config(config_path)

    def test_parses_sections_for_site_json_schema(self):
        sections = import_book.parse_sections(
            '{"sections":[{"heading":"Concept","paragraphs":["Summary"],'
            '"formulas":[],"steps":["Step"],"key_points":["Idea"]}]}'
        )
        self.assertEqual(sections[0]["heading"], "Concept")
        self.assertEqual(sections[0]["paragraphs"], ["Summary"])

    def test_rejects_invalid_sections(self):
        with self.assertRaisesRegex(LLMRequestError, "invalid formulas"):
            import_book.parse_sections(
                '{"sections":[{"heading":"Concept","formulas":"not a list"}]}'
            )

    def test_registers_topic_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            topics_path = Path(directory) / "topics.json"
            topics_path.write_text('["functions"]', encoding="utf-8")
            import_book.register_topic("chapter-one", topics_path)
            import_book.register_topic("chapter-one", topics_path)
            self.assertEqual(
                json.loads(topics_path.read_text(encoding="utf-8")),
                ["functions", "chapter-one"],
            )

    def test_generates_one_section_per_page_chunk(self):
        config = {
            "start_page": 1,
            "end_page": 5,
            "chunk_size": 2,
            "article_id": "chapter-one",
            "title": "Chapter One",
            "overview": "A short overview.",
        }
        pages = [(page, f"Text for page {page}") for page in range(1, 6)]
        model_response = (
            '{"sections":[{"heading":"Concept","paragraphs":["Summary"]}]}'
        )
        with (
            patch.object(import_book, "extract_pdf_pages", return_value=pages),
            patch.object(import_book, "request_completion", return_value=model_response) as request,
        ):
            article = import_book.create_article(config, Path("book.pdf"))

        self.assertEqual(request.call_count, 3)
        self.assertEqual(len(article["sections"]), 3)
        self.assertIn("стр. 1–2", article["sections"][0]["heading"])
        self.assertIn("стр. 5)", article["sections"][2]["heading"])


if __name__ == "__main__":
    unittest.main()
