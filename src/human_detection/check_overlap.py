import numpy as np
from src.human_detection import config


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