import sys
import unittest
from pathlib import Path

import torch.nn as nn

MODULES_DIR = Path(__file__).resolve().parents[1] / "Modules"
sys.path.insert(0, str(MODULES_DIR))

from model import checkpoint_mismatches


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_checkpoint_shape_mismatch_is_reported(self):
        saved_model = nn.Linear(3, 2)
        current_model = nn.Linear(3, 4)

        mismatches = checkpoint_mismatches(current_model, saved_model.state_dict())

        self.assertTrue(any("shape mismatches" in mismatch for mismatch in mismatches))
        self.assertTrue(any("bias" in mismatch for mismatch in mismatches))

    def test_matching_checkpoint_has_no_mismatches(self):
        model = nn.Linear(3, 2)

        self.assertEqual(checkpoint_mismatches(model, model.state_dict()), [])


if __name__ == "__main__":
    unittest.main()
