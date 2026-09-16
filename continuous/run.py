"""One scheduled evaluation cycle. A low score, not ingestion, triggers training."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from continuous.evaluate import load_samples, validate_report


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def training_decision(report, state, *, threshold, min_samples, cooldown, now):
    validate_report(report)
    if report["samples"] < min_samples:
        return "insufficient_samples"
    if report["f1"] >= threshold:
        return "healthy"
    if now - state.get("last_attempt", float("-inf")) < cooldown:
        return "cooldown"
    return "retrain"


def run_cycle(args):
    root = Path(args.state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "cycle.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        state_path = root / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        run_dir = root / uuid.uuid4().hex
        run_dir.mkdir()
        samples, fingerprint = load_samples(args.manifest)
        if len(samples) < args.min_samples:
            result = {"status": "insufficient_samples", "samples": len(samples)}
            save_json(run_dir / "result.json", result)
            return result
        checkpoint = Path(args.checkpoint).resolve()
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        def evaluate(model, name):
            output = run_dir / f"{name}.json"
            subprocess.run([
                sys.executable, "-m", "continuous.evaluate", "--manifest", args.manifest,
                "--checkpoint", str(model), "--output", str(output),
            ], check=True, timeout=args.timeout)
            report = validate_report(json.loads(output.read_text()))
            if report["dataset_sha256"] != fingerprint:
                raise ValueError("Evaluation dataset changed during cycle")
            return report

        baseline = evaluate(checkpoint, "baseline")
        if baseline["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Incumbent checkpoint changed during evaluation")
        decision = training_decision(
            baseline, state, threshold=args.threshold, min_samples=args.min_samples,
            cooldown=args.cooldown_hours * 3600, now=time.time(),
        )
        result = {"status": decision, "baseline": baseline}
        save_json(run_dir / "result.json", result)
        if decision != "retrain":
            return result
        # Persist before starting: a failed or interrupted trainer also observes cooldown.
        state.update(last_attempt=time.time(), run_dir=str(run_dir), status="training")
        save_json(state_path, state)
        try:
            candidate = run_dir / "candidate.pt"
            subprocess.run([
                sys.executable, "train_distillation.py", "--checkpoint", str(checkpoint),
                "--teacher", args.teacher, "--train-csv", args.train_csv,
                "--image-dir", args.image_dir, "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--output", str(candidate),
            ], check=True, timeout=args.timeout)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
                raise ValueError("Incumbent changed during training; rerun evaluation")
            candidate_report = evaluate(candidate, "candidate")
            accepted = (candidate_report["f1"] >= args.threshold
                        and candidate_report["f1"] > baseline["f1"]
                        and candidate_report["f1"] - baseline["f1"] >= args.min_improvement
                        and candidate_report["precision"] >= baseline["precision"]
                        and candidate_report["recall"] >= baseline["recall"])
            result.update(status="candidate_ready" if accepted else "candidate_rejected",
                          candidate=str(candidate), evaluation=candidate_report)
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            raise
        finally:
            state["status"] = result["status"]
            save_json(state_path, state)
            save_json(run_dir / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", default="embedding/weights/best.h5")
    parser.add_argument("--train-csv", default="datasets/train.csv")
    parser.add_argument("--image-dir", default="datasets/card_images_small")
    parser.add_argument("--state-dir", default="runs/continuous")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--cooldown-hours", type=float, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()
    if not (0 < args.threshold <= 1 and 0 <= args.min_improvement <= 1
            and args.min_samples > 0 and args.cooldown_hours >= 0
            and args.epochs > 0 and args.batch_size > 0 and args.timeout > 0):
        parser.error("Invalid evaluation/training limits")
    print(json.dumps(run_cycle(args), indent=2))


if __name__ == "__main__":
    main()
