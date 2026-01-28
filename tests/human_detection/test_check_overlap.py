import cv2
import numpy as np

from src.human_detection.check_overlap import run_safety_monitoring


def get_user_drawn_zone(source=0):
    """
    Opens the camera, freezes the first frame, and lets you click 4 points.
    Returns the binary mask of that zone.
    """
    cap = cv2.VideoCapture(source)

    # Wait for the camera to warm up and grab a valid frame
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        return None

    cap.release()  # Release camera so the main loop can use it later

    # Store clicked points
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw a circle where you clicked to show feedback
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Draw Zone", frame)

    print("\n--- INSTRUCTIONS ---")
    print("1. A window will open with your camera feed.")
    print("2. Click 4 corners to draw your danger zone.")
    print("3. Press ANY KEY to finish drawing and start detection.")

    cv2.imshow("Draw Zone", frame)
    cv2.setMouseCallback("Draw Zone", click_event)
    cv2.waitKey(0)  # Waits indefinitely for a key press
    cv2.destroyAllWindows()

    if len(points) < 3:
        print("Error: You need to click at least 3 points. Defaulting to empty zone.")
        return None

    # Convert points to a binary mask (0s and 1s)
    h, w = frame.shape[:2]
    zone_mask = np.zeros((h, w), dtype=np.uint8)

    # Create the polygon
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(zone_mask, [pts], 1)

    return zone_mask


if __name__ == "__main__":
    # 1. Let the user draw the zone
    print("Initializing camera...")
    my_zone = get_user_drawn_zone(source=0)

    # 2. Run the main safety loop with that zone
    if my_zone is not None:
        run_safety_monitoring(source=0, zone_mask=my_zone)