import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from continuous.index import index_catalog


class Gallery:
    def __init__(self):
        self.id = "gallery"
        self.metadata = {}
        self.rows = {}
    def count(self): return len(self.rows)
    def get(self, ids, include):
        found = [i for i in ids if i in self.rows]
        return {"ids": found, "metadatas": [self.rows[i] for i in found]}
    def modify(self, metadata): self.metadata = metadata
    def update(self, ids, metadatas): self.rows.update(zip(ids, metadatas))
    def upsert(self, ids, embeddings, metadatas): self.update(ids, metadatas)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.teacher = self.root / "teacher.h5"; self.teacher.write_bytes(b"teacher")
        (self.root / "1.jpg").write_bytes(b"one")
        self.rows = [{"id": "1", "card_id": "10", "name": "First", "type": "Effect Monster"}]
        self.collection = Gallery()
        self.encoder = Mock(side_effect=lambda batch: [[0.1, 0.2] for _ in batch])
        self.factory = Mock(return_value=self.encoder)
    def run_index(self, **kwargs):
        return index_catalog(self.collection, self.rows, self.root, self.teacher, self.root, self.factory, **kwargs)

    def test_unchanged_images_never_load_tensorflow_again(self):
        first = self.run_index()
        self.factory.reset_mock()
        second = self.run_index()
        self.assertEqual(first["revision"], second["revision"])
        self.assertEqual(second["encoded"], 0)
        self.factory.assert_not_called()

    def test_new_image_only_is_encoded(self):
        self.run_index(); self.encoder.reset_mock()
        (self.root / "2.jpg").write_bytes(b"two")
        self.rows.append({"id": "2", "type": "Spell Card"})
        self.assertEqual(self.run_index()["encoded"], 1)
        self.assertEqual(len(self.encoder.call_args.args[0]), 1)
        self.assertEqual(self.encoder.call_args.args[0][0][0].name, "2.jpg")

    def test_name_update_does_not_encode_image(self):
        before = self.run_index(); self.factory.reset_mock()
        self.rows[0]["name"] = "Final name"
        after = self.run_index()
        self.assertEqual(after["metadata_updated"], 1)
        self.assertNotEqual(before["revision"], after["revision"])
        self.factory.assert_not_called()

    def test_teacher_change_is_rejected(self):
        self.run_index(); self.teacher.write_bytes(b"different teacher")
        with self.assertRaisesRegex(ValueError, "Teacher changed"): self.run_index()

    def test_missing_image_does_not_trigger_download_or_model_load(self):
        (self.root / "1.jpg").unlink()
        self.assertEqual(self.run_index()["missing_images"], 1)
        self.factory.assert_not_called()

    def test_failed_vector_write_leaves_pending_marker(self):
        self.collection.upsert = Mock(side_effect=RuntimeError("DB unavailable"))
        with self.assertRaises(RuntimeError): self.run_index()
        self.assertTrue(json.loads((self.root / "vector-state.json").read_text())["pending"])

    def test_legacy_gallery_requires_explicit_adoption(self):
        self.collection.rows["1"] = {"id": 1, "name": "First"}
        with self.assertRaisesRegex(ValueError, "adopt-existing"): self.run_index()
        self.assertEqual(self.run_index(adopt_existing=True)["encoded"], 0)
        self.factory.assert_not_called()


if __name__ == "__main__": unittest.main()
