import cv2
import numpy as np
import sys

def main():
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")

    # Create a simple black image to test NumPy/OpenCV integration
    test_image = np.zeros((100, 100, 3), dtype="uint8")
    print("Test image created successfully. Computer Vision environment is active!")

if __name__ == "__main__":
    main()