import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

MODULES_DIR = Path(__file__).resolve().parents[1] / "Modules"
sys.path.insert(0, str(MODULES_DIR))

import train


class TrainerCheckpointTests(unittest.TestCase):
    def test_matching_checkpoint_builds_weighted_criterion_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.pt"
            checkpoint_model = nn.Linear(2, 2)
            torch.save(
                {"model_state": checkpoint_model.state_dict(), "history": {}},
                save_path,
            )

            trainer = train.Trainer(
                device="cpu",
                save_path=save_path,
                skip_if_exists=True,
            )
            train_dataset = Mock()
            train_dataset.y = torch.tensor([0, 0, 0, 1])
            trainer._build_loaders = Mock(
                return_value=(None, None, [], train_dataset)
            )
            expected_metrics = {
                "loss": 0.5,
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "confusion_matrix": [[2, 1], [0, 1]],
            }
            evaluate_mock = Mock(return_value=expected_metrics)

            with (
                patch("train.build_model", return_value=nn.Linear(2, 2)),
                patch("train.evaluate", evaluate_mock),
            ):
                result = trainer.run()

            self.assertEqual(result, expected_metrics)
            criterion = evaluate_mock.call_args.args[2]
            self.assertIsInstance(criterion, nn.CrossEntropyLoss)
            torch.testing.assert_close(
                criterion.weight,
                torch.tensor([2 / 3, 2.0, 1.0, 1.0, 1.0]),
            )


if __name__ == "__main__":
    unittest.main()
