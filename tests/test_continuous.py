import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from continuous.evaluate import load_samples, score_decks, validate_report
from continuous.run import run_cycle, training_decision


def report(f1=0.8, samples=100):
    return dict(f1=f1, samples=samples, cards=100, precision=f1,
                recall=f1, exact_deck_accuracy=f1)


class EvaluationTests(unittest.TestCase):
    def test_duplicate_extra_and_missed_cards_are_penalized(self):
        result = score_decks([[1, 1, 2]], [[1, 2, 3, 4]])
        self.assertAlmostEqual(result["f1"], 4 / 7)
        self.assertEqual(result["exact_deck_accuracy"], 0)

    def test_order_and_numeric_string_ids_do_not_matter(self):
        self.assertEqual(score_decks([[1, 2]], [["2", "1"]])["f1"], 1)

    def test_no_detections_scores_zero(self):
        self.assertEqual(score_decks([[1]], [[]])["f1"], 0)

    def test_nan_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_report(report(float("nan")))

    def test_manifest_rejects_duplicates_and_fingerprints_image_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"first")
            manifest = root / "eval.json"
            row = {"image": "a.png", "card_ids": [1]}
            manifest.write_text(json.dumps([row]))
            _, before = load_samples(manifest)
            (root / "a.png").write_bytes(b"changed")
            _, after = load_samples(manifest)
            self.assertNotEqual(before, after)
            manifest.write_text(json.dumps([row, row]))
            with self.assertRaises(ValueError):
                load_samples(manifest)

    def test_trigger_boundaries(self):
        def decision(value, state=None):
            return training_decision(value, state or {}, threshold=0.95,
                                     min_samples=100, cooldown=60, now=100)
        self.assertEqual(decision(report(0.95)), "healthy")
        self.assertEqual(decision(report(0.8, 99)), "insufficient_samples")
        self.assertEqual(decision(report(), {"last_attempt": 50}), "cooldown")
        self.assertEqual(decision(report(), {"last_attempt": 40}), "retrain")


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "production.pt"
        self.checkpoint.write_bytes(b"production weights")
        (self.root / "deck.png").write_bytes(b"evaluation image")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps([{"image": "deck.png", "card_ids": [1]}]))
        self.args = argparse.Namespace(
            state_dir=str(self.root / "state"), manifest=str(manifest),
            checkpoint=str(self.checkpoint), teacher="teacher.h5", train_csv="train.csv",
            image_dir="images", threshold=0.95, min_improvement=0.01,
            min_samples=1, cooldown_hours=24, epochs=1, batch_size=1, timeout=60,
        )
        self.fingerprint = load_samples(manifest)[1]
        self.baseline, self.candidate = 0.8, 0.98
        self.fail_training = False
        self.calls = []

    def subprocess(self, command, **kwargs):
        self.calls.append(command)
        out = Path(command[command.index("--output") + 1])
        if "continuous.evaluate" in command:
            model = Path(command[command.index("--checkpoint") + 1])
            metrics = report(self.baseline if out.stem == "baseline" else self.candidate, 1)
            metrics.update(dataset_sha256=self.fingerprint,
                           checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
            out.write_text(json.dumps(metrics))
        else:
            if self.fail_training:
                raise subprocess.CalledProcessError(1, command)
            out.write_bytes(b"candidate weights")

    def run_mocked(self):
        with patch("continuous.run.subprocess.run", side_effect=self.subprocess):
            return run_cycle(self.args)

    def test_healthy_model_never_trains(self):
        self.baseline = 0.96
        self.assertEqual(self.run_mocked()["status"], "healthy")
        self.assertEqual(len(self.calls), 1)

    def test_candidate_evaluated_without_overwriting_incumbent(self):
        self.assertEqual(self.run_mocked()["status"], "candidate_ready")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.checkpoint.read_bytes(), b"production weights")
        self.assertEqual(self.run_mocked()["status"], "cooldown")

    def test_regression_is_rejected(self):
        self.candidate = 0.7
        self.assertEqual(self.run_mocked()["status"], "candidate_rejected")

    def test_failure_is_persisted_and_respects_cooldown(self):
        self.fail_training = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_mocked()
        state = json.loads((Path(self.args.state_dir) / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(self.run_mocked()["status"], "cooldown")

    def test_concurrent_run_skips(self):
        root = Path(self.args.state_dir)
        root.mkdir()
        with (root / "cycle.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.run_mocked()["status"], "already_running")
        self.assertEqual(self.calls, [])

    def test_small_dataset_skips_before_loading_model(self):
        self.args.min_samples = 100
        self.assertEqual(self.run_mocked()["status"], "insufficient_samples")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
