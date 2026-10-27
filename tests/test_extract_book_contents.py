import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import extract_book_contents


class ExtractBookContentsTests(unittest.TestCase):
    def test_reads_pdf_and_contents_output_from_import_config(self):
        config = {
            "pdf_path": "books/book.pdf",
            "start_page": 4,
            "end_page": 5,
            "output_filename": "new-topic.json",
            "contents_output_filename": "book_contents.json",
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            pdf_path, output_path, start_page, end_page = (
                extract_book_contents.read_export_config(config_path)
            )

        self.assertEqual(pdf_path, extract_book_contents.PROJECT_DIR / "books" / "book.pdf")
        self.assertEqual(
            output_path,
            extract_book_contents.PROJECT_DIR / "materials" / "book_contents.json",
        )
        self.assertEqual((start_page, end_page), (4, 5))

    def test_rejects_output_filename_that_is_a_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({
                    "pdf_path": "books/book.pdf",
                    "start_page": 4,
                    "end_page": 5,
                    "contents_output_filename": "..\\outside.json",
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "contents_output_filename"):
                extract_book_contents.read_export_config(config_path)

    def test_extracts_chapters_and_subtopics_from_selected_pages(self):
        contents = extract_book_contents.parse_contents_pages([
            """iv
Contents
1
Functions 1
1.1
Functions and Their Graphs 1
1.2
Combining Functions; Shifting and Scaling Graphs 14
Questions to Guide Your Review 33
2
Limits and Continuity 38
2.1
Rates of Change and Tangent Lines to Curves 38
""",
            """3
Derivatives 102
3.1
Tangent Lines and the Derivative at a Point 102
""",
        ])

        self.assertEqual(contents, {
            "topics": [
                {
                    "topic": "1. Functions",
                    "subtopics": [
                        "1.1. Functions and Their Graphs",
                        "1.2. Combining Functions; Shifting and Scaling Graphs",
                    ],
                },
                {
                    "topic": "2. Limits and Continuity",
                    "subtopics": [
                        "2.1. Rates of Change and Tangent Lines to Curves",
                    ],
                },
                {
                    "topic": "3. Derivatives",
                    "subtopics": ["3.1. Tangent Lines and the Derivative at a Point"],
                },
            ],
        })

    def test_extracts_subtopics_written_on_the_same_line_as_their_numbers(self):
        contents = extract_book_contents.parse_contents_pages([
            """10
Infinite Sequences and Series 563
10.1 Sequences 563
10.2 Infinite Series 576
10.10 Applications of Taylor Series 634
11
Parametric Equations and Polar Coordinates 649
11.1 Parametrizations of Plane Curves 649
"""
        ])

        self.assertEqual(contents, {
            "topics": [
                {
                    "topic": "10. Infinite Sequences and Series",
                    "subtopics": [
                        "10.1. Sequences",
                        "10.2. Infinite Series",
                        "10.10. Applications of Taylor Series",
                    ],
                },
                {
                    "topic": "11. Parametric Equations and Polar Coordinates",
                    "subtopics": ["11.1. Parametrizations of Plane Curves"],
                },
            ],
        })

    def test_rejects_pages_without_chapter_entries(self):
        with self.assertRaisesRegex(ValueError, "No chapter entries"):
            extract_book_contents.parse_contents_pages([
                "Contents\nSubject Index\niv\n",
            ])


if __name__ == "__main__":
    unittest.main()
