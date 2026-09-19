import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.camera import imread_unicode, imwrite_unicode


class UnicodePathTests(unittest.TestCase):
    def test_roundtrip_under_non_ascii_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "โฟลเดอร์ไทย"
            folder.mkdir()
            target = folder / "ref.png"
            image = np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)
            self.assertTrue(imwrite_unicode(target, image))
            self.assertTrue(np.array_equal(imread_unicode(target), image))


if __name__ == "__main__":
    unittest.main()
