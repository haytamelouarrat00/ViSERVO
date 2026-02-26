import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import Features

def test_match_xfeat():
    # Create two synthetic images with some shared features
    img1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img1, (100, 100), (200, 200), 255, -1)
    cv2.circle(img1, (400, 300), 50, 255, -1)
    
    img2 = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img2, (110, 110), (210, 210), 255, -1) # Slightly shifted
    cv2.circle(img2, (410, 310), 50, 255, -1) # Slightly shifted
    
    f = Features(img1, img2)
    print("Testing match_xfeat...")
    try:
        kp1, kp2, matches = f.match_xfeat()
        print(f"Success! Found {len(matches)} matches.")
        print(f"Number of keypoints: kp1={len(kp1)}, kp2={len(kp2)}")
        
        if len(matches) > 0:
            print(f"First match: queryIdx={matches[0].queryIdx}, trainIdx={matches[0].trainIdx}")
            print(f"kp1[0] type: {type(kp1[0])}")
            print(f"matches[0] type: {type(matches[0])}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_match_xfeat()
