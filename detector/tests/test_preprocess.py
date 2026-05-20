from django.test import SimpleTestCase

from src.preprocess import clean_text


class CleanTextTests(SimpleTestCase):
    def test_empty_string_returns_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(clean_text(None), "")

    def test_non_string_returns_empty(self):
        self.assertEqual(clean_text(12345), "")
        self.assertEqual(clean_text(3.14), "")
        self.assertEqual(clean_text([]), "")

    def test_lowercases_input(self):
        self.assertNotIn("A", clean_text("ABC"))
        self.assertIn("a", clean_text("ABC"))

    def test_preserves_exclamation_and_question_marks(self):
        out = clean_text("Tin nong! That sao?")
        self.assertIn("!", out)
        self.assertIn("?", out)

    def test_strips_other_punctuation(self):
        out = clean_text("hello, world. @#$%^&*()")
        for ch in ",.@#$%^&*()":
            self.assertNotIn(ch, out)

    def test_collapses_whitespace(self):
        out = clean_text("a    b\t\tc\n\nd")
        self.assertNotIn("  ", out)
        self.assertNotIn("\t", out)
        self.assertNotIn("\n", out)

    def test_keeps_vietnamese_diacritics(self):
        out = clean_text("Tin tức Việt Nam")
        for ch in ["ứ", "ệ"]:
            self.assertIn(ch, out)

    def test_returns_string_type(self):
        self.assertIsInstance(clean_text("abc"), str)
        self.assertIsInstance(clean_text(""), str)
        self.assertIsInstance(clean_text(None), str)
