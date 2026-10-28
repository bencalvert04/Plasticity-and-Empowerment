import unittest
import torch
from input_and_output_directed_information import entropy, jensen_shannon


class JensenShannonTests(unittest.TestCase):
    def test_same_entropy_different_distributions(self):
        p = torch.tensor([.9, .1]).log()
        q = torch.tensor([.1, .9]).log()
        self.assertAlmostEqual(entropy(p), entropy(q), places=6)
        self.assertGreater(jensen_shannon(p, q), .5)
        self.assertAlmostEqual(jensen_shannon(p, q), jensen_shannon(q, p), places=6)

    def test_identical_and_nearly_disjoint(self):
        self.assertAlmostEqual(jensen_shannon(torch.zeros(3), torch.zeros(3)), 0, places=6)
        self.assertAlmostEqual(jensen_shannon(torch.tensor([0., -1000.]), torch.tensor([-1000., 0.])), 1, places=6)


if __name__ == '__main__':
    unittest.main()
