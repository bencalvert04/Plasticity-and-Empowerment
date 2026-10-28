import unittest

from answer_token_scores import answer_span, score_numeric_answer


class PieceTokenizer:
    pieces = ["We compute 2 + 2.\nFinal answer: ", "1", "2", "3", ".", ""]

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


class AnswerScoresTest(unittest.TestCase):
    def test_formats(self):
        for text, expected in [
            ("Intermediate: 15\n#### 1,234", "1,234"),
            (r"Thus \boxed{-2.5}", "-2.5"),
            ("Work\n42", "42"),
            ("The answer is **$18**.", "18"),
            ("Answer: 1/2", "1/2"),
            ("So, James runs 720 meters per week.", "720"),
            (r"Final answer: \( 180 \) meters.", "180"),
            (r"The final answer is \(\boxed{180 \text{ meters}}\).", "180"),
            ("The final answer is: 2280.", "2280"),
            ("Thus, the total distance is **180 meters**.", "180"),
        ]:
            with self.subTest(text=text):
                a, b = answer_span(text)
                self.assertEqual(text[a:b], expected)

    def test_no_guess_for_ambiguous_or_intermediate_number(self):
        for text in ["", "First multiply 3 by 4", "Answer: 3 or 4", "No answer."]:
            self.assertIsNone(answer_span(text))

    def test_exact_generated_token_indices_and_average(self):
        result = score_numeric_answer(PieceTokenizer(), list(range(6)),
                                      [100, 3, -1, 4, 100, 100],
                                      [100, 6, 0, 3, 100, 100], True)
        self.assertEqual(result["answer"], "123")
        self.assertEqual(result["token_indices"], [1, 2, 3])
        self.assertEqual(result["input_bits_per_token"], 2)
        self.assertEqual(result["history_bits_per_token"], 3)

    def test_truncation_and_alignment_errors(self):
        result = score_numeric_answer(PieceTokenizer(), [1], [1], [2], False)
        self.assertIn("did not reach EOS", result["status"])
        with self.assertRaises(ValueError):
            score_numeric_answer(PieceTokenizer(), [1], [], [2], True)


if __name__ == "__main__":
    unittest.main()
