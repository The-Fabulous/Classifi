import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import generate_topic_pages
from tutor_graph import LLMRequestError


class GenerateTopicPagesTests(unittest.TestCase):
    def test_default_run_extracts_configured_pdf_then_generates_pages(self):
        topics = [{
            "topic": "I. INTRO TO C++",
            "subtopics": ["1.1. Computer Program"],
        }]
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "itp_book_contents.json"
            catalog_path.write_text(
                json.dumps({"topics": topics}),
                encoding="utf-8",
            )

            with (
                patch.object(
                    generate_topic_pages.extract_book_contents,
                    "export_book_contents",
                    return_value=(catalog_path, {"topics": topics}),
                ) as export_contents,
                patch.object(
                    generate_topic_pages,
                    "generate_pages",
                    return_value=["book-i-intro-to-c"],
                ) as generate_pages,
                patch.object(sys, "argv", ["generate_topic_pages.py"]),
                redirect_stdout(io.StringIO()),
            ):
                result = generate_topic_pages.main()

        self.assertEqual(result, 0)
        export_contents.assert_called_once_with(generate_topic_pages.DEFAULT_CONFIG)
        generate_pages.assert_called_once_with(
            topics,
            generate_topic_pages.MATERIALS_DIR,
            generate_topic_pages.TOPICS_FILE,
        )

    def test_default_catalog_path_uses_configured_book_contents_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "book_import_config.json"
            config_path.write_text(
                json.dumps({"contents_output_filename": "itp_book_contents.json"}),
                encoding="utf-8",
            )

            catalog_path = generate_topic_pages.default_catalog_path(config_path)

        self.assertEqual(
            catalog_path,
            generate_topic_pages.MATERIALS_DIR / "itp_book_contents.json",
        )

    def test_reads_topic_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contents.json"
            path.write_text(
                json.dumps({
                    "topics": [{
                        "topic": "1. Functions",
                        "subtopics": ["1.1. Functions and Their Graphs"],
                    }],
                }),
                encoding="utf-8",
            )

            self.assertEqual(
                generate_topic_pages.read_topic_catalog(path),
                [{
                    "topic": "1. Functions",
                    "subtopics": ["1.1. Functions and Their Graphs"],
                }],
            )

    def test_generates_article_in_website_schema(self):
        topic = {
            "topic": "1. Functions",
            "subtopics": ["1.1. Functions and Their Graphs"],
        }

        def completion(_messages):
            return json.dumps({
                "overview": "Введение в функции и их графики.",
                "sections": [{
                    "heading": "1.1. Functions and Their Graphs",
                    "paragraphs": ["Функция задаёт соответствие.", "График показывает его."],
                    "formulas": ["y = f(x)"],
                    "steps": [],
                    "key_points": [],
                }],
            })

        article = generate_topic_pages.generate_article(topic, "book-1-functions", completion)

        self.assertEqual(article["id"], "book-1-functions")
        self.assertEqual(article["title"], "1. Functions")
        self.assertEqual(article["sections"][0]["heading"], topic["subtopics"][0])

    def test_rejects_missing_subtopic_explanations(self):
        topic = {
            "topic": "1. Functions",
            "subtopics": ["1.1. Functions and Their Graphs"],
        }

        def completion(_messages):
            return json.dumps({
                "overview": "Введение.",
                "sections": [{
                    "heading": "1.1. Functions and Their Graphs",
                    "paragraphs": ["Только один абзац."],
                }],
            })

        with self.assertRaisesRegex(LLMRequestError, "incomplete explanation"):
            generate_topic_pages.generate_article(topic, "book-1-functions", completion)

    def test_retries_whole_topic_after_invalid_model_json(self):
        topic = {
            "topic": "1. Functions",
            "subtopics": ["1.1. Functions and Their Graphs", "1.2. Combining Functions"],
        }
        calls = 0

        def completion(messages):
            nonlocal calls
            calls += 1
            request = json.loads(messages[1]["content"])
            if len(request["subtopics"]) != 2:
                raise AssertionError("Expected one request for the entire topic.")
            if calls == 1:
                return '{"invalid":'
            return json.dumps({
                "overview": "Введение.",
                "sections": [
                    {
                        "heading": heading,
                        "paragraphs": ["Абзац один.", "Абзац два."],
                    }
                    for heading in request["subtopics"]
                ],
            })

        article = generate_topic_pages.generate_article(topic, "book-1-functions", completion)

        self.assertEqual(calls, 2)
        self.assertEqual(
            [section["heading"] for section in article["sections"]],
            topic["subtopics"],
        )

    def test_retry_explains_missing_paragraphs_to_model(self):
        topic = {
            "topic": "2. Limits and Continuity",
            "subtopics": ["2.1. Limits"],
        }
        calls = 0

        def completion(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps({
                    "overview": "Введение.",
                    "sections": [{
                        "heading": "2.1. Limits",
                        "paragraphs": ["Один абзац."],
                    }],
                })
            self.assertIn("ровно два", messages[-1]["content"])
            self.assertIn("2.1. Limits (1 абз.)", messages[-1]["content"])
            return json.dumps({
                "overview": "Введение.",
                "sections": [{
                    "heading": "2.1. Limits",
                    "paragraphs": ["Первый абзац.", "Второй абзац."],
                }],
            })

        article = generate_topic_pages.generate_article(topic, "book-2-limits", completion)

        self.assertEqual(calls, 2)
        self.assertEqual(len(article["sections"][0]["paragraphs"]), 2)

    def test_sanitizes_latex_backslashes_before_json_parse(self):
        response = json.dumps({
            "overview": "Введение.",
            "sections": [{
                "heading": "1.1. Functions and Their Graphs",
                "paragraphs": ["Формула: \\lim_{x\\to 0} \\frac{f(x)-f(0)}{x}", "Смысл: это предел."],
                "formulas": ["\\frac{a}{b}"],
                "steps": [],
                "key_points": [],
            }],
        })

        result = generate_topic_pages.parse_model_page(response, "1. Functions", ["1.1. Functions and Their Graphs"])

        self.assertEqual(result[1][0]["heading"], "1.1. Functions and Their Graphs")

    def test_writes_pages_and_registers_them_without_duplicates(self):
        topics = [{
            "topic": "1. Functions",
            "subtopics": ["1.1. Functions and Their Graphs"],
        }]

        def completion(_messages):
            return json.dumps({
                "overview": "Введение.",
                "sections": [{
                    "heading": "1.1. Functions and Their Graphs",
                    "paragraphs": ["Абзац один.", "Абзац два."],
                }],
            })

        with tempfile.TemporaryDirectory() as directory:
            materials_dir = Path(directory) / "materials"
            materials_dir.mkdir()
            topics_file = materials_dir / "topics.json"
            topics_file.write_text('["existing"]', encoding="utf-8")

            for _ in range(2):
                generate_topic_pages.generate_pages(
                    topics,
                    materials_dir,
                    topics_file,
                    completion,
                )

            registered = json.loads(topics_file.read_text(encoding="utf-8"))
            article = json.loads(
                (materials_dir / "book-1-functions.json").read_text(encoding="utf-8")
            )

        self.assertEqual(registered, ["existing", "book-1-functions"])
        self.assertEqual(article["sections"][0]["paragraphs"], ["Абзац один.", "Абзац два."])

    def test_creates_missing_topic_index_after_generating_pages(self):
        topics = [{
            "topic": "I. INTRO TO C++",
            "subtopics": ["1.1. Computer Program"],
        }]

        def completion(_messages):
            return json.dumps({
                "overview": "Краткое введение.",
                "sections": [{
                    "heading": "1.1. Computer Program",
                    "paragraphs": ["Первый абзац.", "Второй абзац."],
                }],
            })

        with tempfile.TemporaryDirectory() as directory:
            materials_dir = Path(directory) / "materials"
            materials_dir.mkdir()
            topics_file = materials_dir / "topics.json"

            article_ids = generate_topic_pages.generate_pages(
                topics,
                materials_dir,
                topics_file,
                completion,
            )

            self.assertEqual(
                json.loads(topics_file.read_text(encoding="utf-8")),
                article_ids,
            )


if __name__ == "__main__":
    unittest.main()
