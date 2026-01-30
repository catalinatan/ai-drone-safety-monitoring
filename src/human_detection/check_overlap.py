import cv2
import numpy as np
from human_detection import config
from human_detection.detector import HumanDetector


# --- FUNCTION 1: THE LOGIC ---
def check_danger_zone_overlap(person_masks, zone_mask):
    """
    Checks if any person overlaps with the danger zone mask.
    Returns: (is_alarm, list_of_danger_masks)
    """
    is_alarm = False
    danger_masks = []

    # If no zone is defined, we can't calculate safety
    if zone_mask is None:
        return False, []

    for person_mask in person_masks:
        # Optimization: Skip tiny noise
        if np.sum(person_mask) < config.MIN_PERSON_AREA_PIXELS:
            continue

        # Intersection: Where (Person == 1) AND (Zone == 1)
        intersection = np.bitwise_and(person_mask, zone_mask)
        intersection_area = np.sum(intersection)
        person_area = np.sum(person_mask)

        if person_area == 0: continue

        overlap_ratio = intersection_area / person_area

        if overlap_ratio >= config.DANGER_ZONE_OVERLAP_THRESHOLD:
            is_alarm = True
            danger_masks.append(person_mask)

    return is_alarm, danger_masks


# --- FUNCTION 2: THE VISUALIZATION ---
def draw_danger_annotations(frame, danger_masks):
    """
    Draws RED boxes and '!' warnings around people in danger.
    """
    annotated_frame = frame.copy()

    for mask in danger_masks:
        # Find the bounding box of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue

        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Draw Box
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Draw '!' Box
        cv2.rectangle(annotated_frame, (x, y - 40), (x + 40, y), (0, 0, 255), -1)
        cv2.putText(annotated_frame, "!", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Draw Label
        cv2.putText(annotated_frame, "DANGER", (x + 50, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated_frame


# --- FUNCTION 3: THE RUNNER LOOP ---
def run_safety_monitoring(source=0, zone_mask=None):
    """
    Main loop that captures video, detects humans, checks safety,
    and displays the result.
    """
    cap = cv2.VideoCapture(source)
    detector = HumanDetector()

    # If no zone mask is provided, create a dummy one (top half of screen)
    # This ensures the code doesn't crash if you forget to pass one.
    if zone_mask is None:
        print("WARNING: No Zone Mask provided. Using Top-Half of screen as danger zone.")
        # We need to wait for the first frame to know the size
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            zone_mask = np.zeros((h, w), dtype=np.uint8)
            zone_mask[0:h // 2, :] = 1  # Top half is dangerous

    print("Starting Safety Monitoring... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. AI Detection
        masks = detector.get_masks(frame)

        # 2. Logic Check
        alarm_active, danger_masks = check_danger_zone_overlap(masks, zone_mask)

        # 3. Visualization
        if alarm_active:
            # Draw RED boxes on danger people
            display_frame = draw_danger_annotations(frame, danger_masks)

            # Optional: Add "ALARM" text to the whole screen
            cv2.putText(display_frame, "ALARM ACTIVE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            # Safe: Just show original frame (or draw green boxes if you want)
            display_frame = frame

        # 4. Show Zone (Optional Debugging: Lightly overlay the zone so you can see it)
        # Make the danger zone appear as a faint red tint
        if zone_mask is not None:
            # Create a red overlay
            red_overlay = np.zeros_like(frame)
            red_overlay[:, :] = (0, 0, 255)
            # Apply it only where the zone is
            display_frame = np.where(zone_mask[..., None] == 1,
                                     cv2.addWeighted(display_frame, 0.7, red_overlay, 0.3, 0),
                                     display_frame)

        cv2.imshow("Safety Monitoring System", display_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()