"""Run with training dependencies: python -m unittest discover -s tests/ml -v."""
import unittest
from unittest.mock import patch, Mock
import numpy as np
import torch
from models.torch import Detector
from loss.torch.detection_loss import v8DetectionLoss
from data.dataset.torch.deck_dataset import DecklistDataset


class TrainingSmokeTests(unittest.TestCase):
    def test_parallel_load_order_keeps_pendulum_labels_aligned(self):
        dataset = DecklistDataset.__new__(DecklistDataset)
        dataset.deck_data = [[("normal.jpg", False), ("pendulum.jpg", True)]]
        dataset.pixelate_scale = dataset.zoom_scale = (1, 1)
        dataset.normal_pos = np.array([0.1, 0.2, 0.8, 0.9])
        dataset.pendulum_pos = np.array([0.2, 0.3, 0.7, 0.8])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        dataset.image_loader = Mock()
        dataset.image_loader.run.return_value = [image.copy(), image.copy()]
        dataset.image_loader.get_file_names.return_value = ["pendulum.jpg", "normal.jpg"]
        module = "data.dataset.torch.deck_dataset."
        with patch(module + "random_pixelate", side_effect=lambda x, *args: x), \
             patch(module + "random_zoom_transition", side_effect=lambda x, *args: (x, 1, np.zeros(2))), \
             patch(module + "make_deck_image", return_value=(image, np.zeros((2, 2)))), \
             patch(module + "make_square_shape", return_value=(image, 1, np.zeros(2))):
            batch = dataset[0]
        np.testing.assert_allclose(batch["xyxy"][0], dataset.pendulum_pos)
        np.testing.assert_allclose(batch["xyxy"][1], dataset.normal_pos)

    def test_foreground_embedding_head_receives_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = Detector().train()
        predictions = model(torch.rand(2, 3, 64, 64))
        batch = {
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.zeros(2),
            "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.6]] * 2),
            "embedding": torch.nn.functional.normalize(torch.rand(2, 256), dim=-1),
        }
        loss, components, mask = v8DetectionLoss(model.layer22, torch.device("cpu"))(
            predictions, batch
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(mask.any())
        self.assertGreater(components[3].item(), 0)
        loss.backward()
        grads = [p.grad for name, p in model.named_parameters() if "embedding" in name]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_no_foreground_does_not_create_nan_embedding_loss(self):
        torch.set_num_threads(1)
        model = Detector().train()
        predictions = model(torch.rand(2, 3, 64, 64))
        batch = {"batch_idx": torch.empty(0), "cls": torch.empty(0),
                 "bboxes": torch.empty(0, 4), "embedding": torch.empty(0, 256)}
        loss, components, mask = v8DetectionLoss(model.layer22, torch.device("cpu"))(
            predictions, batch
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(components[3].item(), 0)
        self.assertFalse(mask.any())
        loss.backward()


if __name__ == "__main__":
    unittest.main()
