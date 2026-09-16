"""One scheduled evaluation cycle. A low score, not ingestion, triggers training."""
import argparse
import fcntl
import hashlib
import csv
import os
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
        from continuous.index import file_hash
        cache = state.get("files", {})
        samples, fingerprint = load_samples(args.manifest)
        if len(samples) < args.min_samples:
            return {"status": "insufficient_samples", "samples": len(samples)}
        checkpoint = Path(args.checkpoint).resolve()
        checkpoint_hash = file_hash(checkpoint, cache)
        vector_state_path = root / "vector-state.json"
        vectors = json.loads(vector_state_path.read_text())
        external_path = root / "external-index.json"
        external = json.loads(external_path.read_text()) if external_path.exists() else {}
        if external.get("pending") or external.get("revision") != vectors.get("external_revision"):
            raise ValueError("Gallery import changed; run incremental indexing before evaluation")
        if vectors.get("pending") or not vectors.get("revision"):
            raise ValueError("Gallery indexing is incomplete; finish indexing before evaluation")
        # Cheap source hashing ensures an inference/preprocessing code change invalidates the cache.
        code = hashlib.sha256()
        repo = Path(__file__).resolve().parents[1]
        for path in sorted([repo / "server.py", repo / "requirements.txt", repo / "requirements-training.txt", *repo.glob("models/**/*.py"),
                            *repo.glob("data/preprocess/**/*.py"), *repo.glob("utils/*.py"),
                            *repo.glob("db/*.py"), *repo.glob("continuous/*.py")]):
            code.update(path.read_bytes())
        evaluation_key = hashlib.sha256(json.dumps([
            checkpoint_hash, fingerprint, vectors["revision"], vectors["collection_id"], code.hexdigest(),
            {key: os.getenv(key, "") for key in ("chroma_mode", "chroma_path", "chroma_collection", "host", "chroma_port", "TOP_K")},
            (repo / ".env").read_text() if (repo / ".env").exists() else "",
        ], sort_keys=True).encode()).hexdigest()
        same_evaluation = state.get("evaluation_key") == evaluation_key and not getattr(args, "force_evaluation", False)
        run_dir = root / uuid.uuid4().hex

        child_env = dict(os.environ)
        child_env.setdefault("OMP_NUM_THREADS", "2")
        child_env.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        child_env.setdefault("TF_NUM_INTEROP_THREADS", "1")

        def evaluate(model, name):
            output = run_dir / f"{name}.json"
            subprocess.run([
                sys.executable, "-m", "continuous.evaluate", "--manifest", args.manifest,
                "--checkpoint", str(model), "--output", str(output),
            ], check=True, timeout=args.timeout, env=child_env)
            report = validate_report(json.loads(output.read_text()))
            if report["dataset_sha256"] != fingerprint:
                raise ValueError("Evaluation dataset changed during cycle")
            return report

        if same_evaluation:
            baseline = validate_report(state["baseline"])
        else:
            run_dir.mkdir()
            baseline = evaluate(checkpoint, "baseline")
            if baseline["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Incumbent checkpoint changed during evaluation")
            state.update(evaluation_key=evaluation_key, baseline=baseline, files=cache)
            save_json(state_path, state)
        decision = training_decision(
            baseline, state, threshold=args.threshold, min_samples=args.min_samples,
            cooldown=args.cooldown_hours * 3600, now=time.time(),
        )
        result = {"status": "unchanged" if same_evaluation and decision == "healthy" else decision,
                  "baseline": baseline, "evaluation_cached": same_evaluation}
        if decision != "retrain":
            if run_dir.exists(): save_json(run_dir / "result.json", result)
            return result
        with open(args.train_csv, newline="") as stream:
            training_rows = list(csv.DictReader(stream))
        training_key = hashlib.sha256(json.dumps([
            evaluation_key, file_hash(args.teacher, cache), file_hash(args.train_csv, cache),
            [file_hash(Path(args.image_dir) / (row["id"] + ".jpg"), cache) for row in training_rows],
            args.epochs, args.batch_size, args.threshold, args.min_improvement,
        ]).encode()).hexdigest()
        if same_evaluation and state.get("completed_training_key") == training_key:
            return {**result, "status": "unchanged"}
        run_dir.mkdir(exist_ok=True)
        save_json(run_dir / "baseline.json", baseline)
        state["files"] = cache
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
            ], check=True, timeout=args.timeout, env=child_env)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
                raise ValueError("Incumbent changed during training; rerun evaluation")
            current_vectors = json.loads(vector_state_path.read_text())
            if current_vectors.get("pending") or current_vectors.get("revision") != vectors["revision"]:
                raise ValueError("Gallery changed during cycle")
            candidate_report = evaluate(candidate, "candidate")
            accepted = (candidate_report["f1"] >= args.threshold
                        and candidate_report["f1"] > baseline["f1"]
                        and candidate_report["f1"] - baseline["f1"] >= args.min_improvement
                        and candidate_report["precision"] >= baseline["precision"]
                        and candidate_report["recall"] >= baseline["recall"])
            result.update(status="candidate_ready" if accepted else "candidate_rejected",
                          candidate=str(candidate), evaluation=candidate_report)
            state["completed_training_key"] = training_key
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
    parser.add_argument("--force-evaluation", action="store_true", help="Bypass the unchanged-input evaluation cache")
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
