import airsim
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R

def get_feet_from_mask(mask):
    """
    Analyzes a binary mask (0=bg, 1=human) to find the bottom-center pixel.
    Returns: (x, y) tuple or None if mask is empty.
    """
    # Find all indices where the mask is > 0 (human pixels)
    rows, cols = np.where(mask > 0)
    
    if len(rows) == 0:
        return None
    
    # The lowest point (largest Y value) corresponds to the feet
    max_y = np.max(rows)
    
    # Get all X coordinates at that lowest Y level
    xs_at_bottom = cols[rows == max_y]
    
    # Average them to find the horizontal center of the feet
    center_x = int(np.mean(xs_at_bottom))
    
    return (center_x, max_y)


def unreal_to_airsim(unreal_x, unreal_y, unreal_z):
    """
    Converts Unreal Engine coordinates (cm, Z-Up) to AirSim coordinates (m, Z-Down/NED).
    
    Args:
        unreal_x, unreal_y, unreal_z: The Location values from the Unreal Details Panel (in cm).
        
    Returns:
        dict: A dictionary suitable for pasting into settings.json
    """
    # 1. Scale: Convert Centimeters to Meters
    x_m = unreal_x / 100.0
    y_m = unreal_y / 100.0
    z_m = unreal_z / 100.0
    
    # 2. Coordinate Swap: Unreal Z is UP (+), AirSim Z is DOWN (+)
    # If an object is 10m high in Unreal, it is -10m (above ground) in AirSim.
    airsim_z = -z_m
    
    return {"X": x_m, "Y": y_m, "Z": airsim_z}


def get_coords_from_ai_depth(client, camera_name, pixel_x, pixel_y, img_w, img_h, ai_depth_val, ground_scale_factor=1.0):
    """
    Converts a pixel + AI Depth value into a 3D World Coordinate.
    
    Args:
        ai_depth_val: The raw value from your depth estimation model (e.g. 0.45)
        ground_scale_factor: The multiplier to convert AI units to Real Meters.
    """
    
    # 1. Convert Relative AI Depth to Real Meters
    # (If using a 'Metric' model like ZoeDepth, factor is 1.0)
    real_dist = ai_depth_val * ground_scale_factor

    # 2. Get Camera Physics (Intrinsics)
    # Assuming standard 90 degree FOV
    fov_rad = np.deg2rad(90.0)
    focal_len = (img_w / 2) / np.tan(fov_rad / 2)

    # 3. Inverse Projection (Pixel -> Camera Frame 3D Vector)
    # Origin is image center
    c_x = img_w / 2
    c_y = img_h / 2

    # In AirSim Camera Frame: X=Forward, Y=Right, Z=Down
    # We calculate the vector components based on the pixel offset
    x_local = real_dist
    y_local = real_dist * (pixel_x - c_x) / focal_len
    z_local = real_dist * (pixel_y - c_y) / focal_len

    point_local = np.array([x_local, y_local, z_local])

    # 4. Get Camera Pose (Extrinsics)
    info = client.simGetCameraInfo(camera_name)
    cam_pos = info.pose.position
    cam_orient = info.pose.orientation

    # 5. Rotate Local Vector to World Orientation
    r = R.from_quat([cam_orient.x_val, cam_orient.y_val, cam_orient.z_val, cam_orient.w_val])
    point_world_vec = r.as_matrix() @ point_local

    # 6. Add Camera Position to get Final Coordinate
    final_x = cam_pos.x_val + point_world_vec[0]
    final_y = cam_pos.y_val + point_world_vec[1]
    final_z = cam_pos.z_val + point_world_vec[2]

    return airsim.Vector3r(final_x, final_y, final_z)


def get_coords_from_lite_mono(client, camera_name, pixel_x, pixel_y, img_w, img_h, ai_depth_val, cctv_height_meters, vehicle_name=""):
    """
    Converts Lite-Mono relative depth (0.0 to 1.0) into World Coordinates using ground plane intersection.

    This method works for cameras at ANY angle by:
    1. Computing the ray direction from camera through the pixel
    2. Intersecting that ray with the ground plane (Z=0)
    3. Using relative depth to weight between close/far possibilities

    Args:
        client: AirSim client
        camera_name: Name of the camera
        pixel_x, pixel_y: Pixel coordinates of the target
        img_w, img_h: Image dimensions
        ai_depth_val: Relative depth from LiteMono (0.0 = close, 1.0 = far)
        cctv_height_meters: Camera height above ground (used as fallback)
        vehicle_name: Vehicle the camera is attached to (empty for external cameras)
    """

    # --- STEP 1: GET CAMERA INFO ---
    info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)
    cam_pos = info.pose.position
    cam_orient = info.pose.orientation

    # --- STEP 2: COMPUTE RAY DIRECTION IN CAMERA FRAME ---
    # Use pinhole camera model to get normalized ray direction
    fov_rad = np.deg2rad(90.0)
    focal_len = (img_w / 2) / np.tan(fov_rad / 2)

    c_x = img_w / 2
    c_y = img_h / 2

    # Normalized ray direction in camera frame (X=Forward, Y=Right, Z=Down)
    ray_x = 1.0
    ray_y = (pixel_x - c_x) / focal_len
    ray_z = (pixel_y - c_y) / focal_len

    # Normalize to unit vector
    ray_cam = np.array([ray_x, ray_y, ray_z])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # --- STEP 3: ROTATE RAY TO WORLD FRAME ---
    r = R.from_quat([cam_orient.x_val, cam_orient.y_val, cam_orient.z_val, cam_orient.w_val])
    ray_world = r.as_matrix() @ ray_cam

    # --- STEP 4: INTERSECT RAY WITH GROUND PLANE (Z=0) ---
    # Ray equation: P = cam_pos + t * ray_world
    # Ground plane: Z = 0
    # Solve for t: cam_pos.z + t * ray_world[2] = 0

    ground_z = 0.0  # Assuming ground is at Z=0 in NED frame

    if abs(ray_world[2]) < 0.001:
        # Ray is nearly horizontal, can't intersect ground reliably
        # Fall back to heuristic scaling
        distance = ai_depth_val * (cctv_height_meters * 2.5)
    else:
        # Calculate intersection parameter
        t_ground = (ground_z - cam_pos.z_val) / ray_world[2]

        if t_ground < 0:
            # Ray points away from ground (camera looking up)
            # Fall back to heuristic
            distance = ai_depth_val * (cctv_height_meters * 2.5)
        else:
            # We have a valid ground intersection!
            # Use relative depth to scale: 0.0 = nearby, 1.0 = ground intersection
            # For feet detection, we want the ground point, so use higher weight for ai_depth
            min_distance = max(1.0, cctv_height_meters * 0.5)  # Minimum realistic distance
            max_distance = t_ground  # Maximum = ground intersection

            # Scale based on relative depth (exponential for better distribution)
            distance = min_distance + (max_distance - min_distance) * (ai_depth_val ** 0.5)

    # --- STEP 5: COMPUTE FINAL WORLD POSITION ---
    point_world = np.array([
        cam_pos.x_val + distance * ray_world[0],
        cam_pos.y_val + distance * ray_world[1],
        cam_pos.z_val + distance * ray_world[2]
    ])

    # Clamp Z to ground level (feet should be on ground)
    point_world[2] = min(point_world[2], 0.0)  # In NED, 0 is ground, negative is above

    return airsim.Vector3r(point_world[0], point_world[1], point_world[2])


# =========================================================
# 3. NEW HELPER: MANUAL ANNOTATION
# =========================================================
def get_user_defined_danger_zone(client, cctv_name, vehicle_name=""):
    """
    Captures one image from AirSim and lets the user draw the danger zone.
    Returns: A binary mask (1=Danger, 0=Safe)

    Args:
        client: AirSim client instance
        cctv_name: Name of the camera
        vehicle_name: Name of the vehicle (drone) that has the camera
    """
    print("[SETUP] Capturing image for Danger Zone annotation...")

    # Get 1 frame from the specified vehicle's camera
    responses = client.simGetImages([
        airsim.ImageRequest(cctv_name, airsim.ImageType.Scene, False, False)
    ], vehicle_name=vehicle_name)

    if not responses:
        print("[ERROR] Could not get image from CCTV.")
        print(f"[ERROR] Camera: {cctv_name}, Vehicle: {vehicle_name}")
        sys.exit(1)
        
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    
    print("[SETUP] instructions:")
    print("  1. Click points to define the polygon (Danger Zone).")
    print("  2. Press 'SPACE' or 'ENTER' to finish and close.")
    print("  3. Press 'c' to clear and restart.")
    
    # Normalize for OpenCV display if needed
    display_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Create a window and let user draw ROI
    # We use a polygon selector loop
    pts = []
    
    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow("Define Danger Zone")
    cv2.setMouseCallback("Define Danger Zone", draw_polygon)

    while True:
        temp_img = display_img.copy()
        
        # Draw lines between points
        if len(pts) > 0:
            cv2.polylines(temp_img, [np.array(pts)], isClosed=False, color=(0, 0, 255), thickness=2)
            for pt in pts:
                cv2.circle(temp_img, pt, 4, (0, 0, 255), -1)
                
        cv2.imshow("Define Danger Zone", temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Enter/Space to confirm
        if key == 13 or key == 32: 
            break
        # 'c' to clear
        if key == ord('c'):
            pts = []

    cv2.destroyWindow("Define Danger Zone")
    
    # Create the binary mask from the polygon
    mask = np.zeros((response.height, response.width), dtype=np.uint8)
    if len(pts) > 2:
        cv2.fillPoly(mask, [np.array(pts)], 1)
        print("[SETUP] Danger Zone defined successfully.")
        return mask
    else:
        print("[WARNING] No valid polygon drawn. Danger Zone is empty.")
        return mask