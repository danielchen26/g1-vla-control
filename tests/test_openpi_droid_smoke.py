import sys
from pathlib import Path
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpi_droid_smoke import (
    MockDroidPolicy, make_synthetic_droid_observation, run_smoke_audit,
)


class InvalidPolicy(MockDroidPolicy):
    def infer(self, observation):
        response = super().infer(observation)
        response["actions"][0, 0] = np.nan
        return response


class SlowPolicy(MockDroidPolicy):
    def infer(self, observation):
        time.sleep(0.05)
        return super().infer(observation)


class OpenPiDroidSmokeTests(unittest.TestCase):
    def test_official_droid_shaped_observation_is_deterministic(self):
        first = make_synthetic_droid_observation()
        second = make_synthetic_droid_observation()
        self.assertEqual(first["observation/exterior_image_1_left"].shape, (224, 224, 3))
        self.assertEqual(first["observation/wrist_image_left"].dtype, np.uint8)
        self.assertEqual(first["observation/joint_position"].shape, (7,))
        self.assertEqual(first["observation/gripper_position"].shape, (1,))
        np.testing.assert_array_equal(
            first["observation/exterior_image_1_left"],
            second["observation/exterior_image_1_left"],
        )

    def test_mock_audit_passes_without_claiming_neural_vla(self):
        report = run_smoke_audit(
            MockDroidPolicy(), make_synthetic_droid_observation,
            calls=4, warmup_calls=1, evidence_mode="deterministic_mock_transport",
        )
        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["valid_calls"], 4)
        self.assertEqual(report["summary"]["unique_action_shapes"], [[10, 8]])
        self.assertFalse(report["neural_vla_claimed"])
        self.assertFalse(report["g1_execution_enabled"])
        self.assertFalse(report["g1_action_compatible"])

    def test_inference_timeout_fails_closed(self):
        report = run_smoke_audit(
            SlowPolicy(), make_synthetic_droid_observation,
            calls=1, warmup_calls=0, call_timeout_ms=5,
            evidence_mode="deterministic_mock_transport",
        )
        self.assertFalse(report["summary"]["passed"])
        self.assertIn("TimeoutError", report["calls"][0]["error"])

    def test_nonfinite_actions_fail_closed(self):
        report = run_smoke_audit(
            InvalidPolicy(), make_synthetic_droid_observation,
            calls=2, warmup_calls=0, evidence_mode="deterministic_mock_transport",
        )
        self.assertFalse(report["summary"]["passed"])
        self.assertEqual(report["summary"]["valid_calls"], 0)


if __name__ == "__main__":
    unittest.main()
