"""Incremental gallery indexing. TensorFlow is loaded only when vectors are missing/stale."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

from continuous.run import save_json


def file_hash(path, cache):
    path = Path(path).resolve()
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    previous = cache.get(str(path), {})
    if previous.get("stamp") != stamp:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        previous = {"stamp": stamp, "sha256": digest}
        cache[str(path)] = previous
    return previous["sha256"]


def read_catalog(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        if not row.get("id", "").isdigit() or not row.get("type") or row["id"] in seen:
            raise ValueError("Catalog requires unique numeric image id and type columns")
        seen.add(row["id"])
    if not rows:
        raise ValueError("Catalog is empty")
    return rows


def index_catalog(collection, rows, image_dir, teacher, root, embed_factory, *, adopt_existing=False, batch_size=8):
    root = Path(root)
    state_path = root / "vector-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    cache = state.get("files", {})
    teacher_hash = file_hash(teacher, cache)
    known_teacher = state.get("teacher_sha256") if state.get("collection_id") == str(collection.id) else None
    if known_teacher and known_teacher != teacher_hash:
        raise ValueError("Teacher changed; rebuild a separate collection instead of mixing embedding spaces")
    if not known_teacher and collection.count() and not adopt_existing:
        raise ValueError("Existing gallery needs --adopt-existing after verifying the matching teacher")
    identity = str(collection.id)
    old_revision = state.get("revision")
    changed = state.get("pending", False) or state.get("collection_id") != identity or not old_revision
    embed = None
    encoded = updated = missing = 0
    marked_pending = False
    external_path = root / "external-index.json"
    external = json.loads(external_path.read_text()) if external_path.exists() else {}
    if external.get("pending"):
        raise ValueError("Snapshot import is incomplete; finish import before indexing")
    external_revision = external.get("revision")
    changed = changed or state.get("external_revision") != external_revision

    def mark_pending():
        nonlocal changed, marked_pending
        changed = True
        if not marked_pending:
            save_json(state_path, {**state, "pending": True, "collection_id": identity, "teacher_sha256": teacher_hash})
            marked_pending = True

    if known_teacher is None:
        mark_pending()
        # Bind the teacher to this collection in local state; do not change HNSW metadata.

    for offset in range(0, len(rows), 200):
        chunk = rows[offset:offset + 200]
        found = collection.get(ids=[r["id"] for r in chunk], include=["metadatas"])
        existing = dict(zip(found["ids"], found["metadatas"]))
        pending = []
        for row in chunk:
            path = Path(image_dir) / (row["id"] + ".jpg")
            if not path.is_file():
                missing += 1
                continue  # ingestion owns downloads; never fetch from upstream here
            digest = file_hash(path, cache)
            desired = {"id": int(row["id"]), "card_id": int(row.get("card_id") or row["id"]),
                       "name": row.get("name") or row.get("en_name") or row["id"],
                       "type": row["type"], "image_sha256": digest, "teacher_sha256": teacher_hash}
            previous = existing.get(row["id"])
            # Explicit adoption trusts existing vectors; future image/type changes are tracked.
            adopted = (previous is not None and adopt_existing and "image_sha256" not in previous)
            same_vector = previous is not None and (
                adopted or all(previous.get(k) == desired[k] for k in ("image_sha256", "teacher_sha256", "type")))
            if same_vector:
                merged = {**previous, **desired}
                if merged != previous:
                    mark_pending()
                    collection.update(ids=[row["id"]], metadatas=[merged])
                    updated += 1
            else:
                pending.append((row["id"], path, desired))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if embed is None:
                embed = embed_factory(teacher)
            vectors = embed([(path, meta["type"]) for _, path, meta in batch])
            mark_pending()
            collection.upsert(ids=[i for i, _, _ in batch], embeddings=vectors,
                              metadatas=[meta for _, _, meta in batch])
            encoded += len(batch)
    count = collection.count()
    changed = changed or state.get("count") != count
    revision = uuid.uuid4().hex if changed else old_revision
    save_json(state_path, {"external_revision": external_revision, "pending": False, "revision": revision, "collection_id": identity,
                          "count": count, "teacher_sha256": teacher_hash, "files": cache})
    return {"encoded": encoded, "metadata_updated": updated, "missing_images": missing, "revision": revision}


def tensorflow_embedder(teacher):
    import cv2
    import numpy as np
    import tensorflow as tf
    from models.tf import EmbeddingModel
    from data.preprocess.tf import EmbeddingPreprocessor
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = EmbeddingModel(model_path=str(teacher))
    preprocess = EmbeddingPreprocessor()

    def embed(batch):
        inputs = []
        for path, card_type in batch:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Invalid card image: {path}")
            inputs.append(preprocess(image[:, :, ::-1].copy(), "pendulum" in card_type.lower()))
        vectors = model(tf.stack(inputs)).numpy()
        if not np.isfinite(vectors).all():
            raise ValueError("Non-finite gallery vectors")
        return vectors.tolist()
    return embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--state-dir", default="runs/continuous")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    root = Path(args.state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "cycle.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another index/evaluation cycle is running")
        from db.chroma_db import ChromaDBConnection
        print(json.dumps(index_catalog(ChromaDBConnection().collection, read_catalog(args.catalog_csv),
            args.image_dir, args.teacher, root, tensorflow_embedder,
            adopt_existing=args.adopt_existing, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
