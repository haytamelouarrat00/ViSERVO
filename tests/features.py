import unittest
import numpy as np
import cv2
from src.features import Features

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic images with a white square for feature detection
        self.img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(self.img, (50, 50), (150, 150), 255, -1)
        self.features = Features(self.img, self.img)

    def test_pipeline_integration(self):
        # Test basic flow: SIFT matching
        kp1, kp2, matches = self.features.match_sift()
        self.assertGreater(
            len(matches), 0, "SIFT should find matches on identical images"
        )
        self.assertGreater(len(kp1), 0)
        self.assertGreater(len(kp2), 0)

    def test_xfeat_match_runs(self):
        # Optional dependency: skip if XFeat isn't available in this env.
        try:
            kp1, kp2, matches = self.features.match_xfeat(top_k=512)
        except ImportError:
            self.skipTest("XFeat not available in current environment")

        self.assertEqual(len(kp1), len(kp2))
        self.assertLessEqual(len(matches), len(kp1))


if __name__ == "__main__":
    unittest.main()
