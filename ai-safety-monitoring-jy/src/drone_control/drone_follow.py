import airsim
import time

# 1. Connect
client = airsim.MultirotorClient()
client.confirmConnection()

# 2. Drones and Offsets [X, Y, Z]
drones = {
    "CCTV_Drone1": [15, 10, -12], 
    "CCTV_Drone2": [15, -10, -12],
    "CCTV_Drone3": [-25, 0, -15],
    "CCTV_Drone4": [0, 0, -30]
}

# 3. Arm and Takeoff

for name in drones:
    client.enableApiControl(True, vehicle_name=name)
    client.armDisarm(True, vehicle_name=name)
    client.takeoffAsync(vehicle_name=name)

ship_name = "chalutier_Blueprint"
print(f"System Active: Drones following {ship_name}...")

try:
    while True:
        ship_pose = client.simGetObjectPose(ship_name)
        pos = ship_pose.position

        # Verify the ship isn't returning NaN
        import math
        if math.isnan(pos.x_val):
            print("Waiting for ship to initialize (NaN detected)...", end="\r")
            time.sleep(0.5)
            continue

        print(f"Tracking Ship at X: {pos.x_val:.2f}, Y: {pos.y_val:.2f}", end="\r")

        for name, offset in drones.items():
            # Calculate position + offset
            target_pos = airsim.Vector3r(pos.x_val + offset[0], 
                                         pos.y_val + offset[1], 
                                         pos.z_val + offset[2])
            
            # Use moveToPosition to handle the 'Look At' automatically
            # drivetrain=ForwardOnly makes the drone face the direction it's moving
            client.moveToPositionAsync(target_pos.x_val, target_pos.y_val, target_pos.z_val, 
                                     velocity=5, vehicle_name=name,
                                     drivetrain=airsim.DrivetrainType.ForwardOnly, 
                                     yaw_mode=airsim.YawMode(False, 0))
            
        time.sleep(0.05) # 20 updates per second

except KeyboardInterrupt:
    print("\nUser stopped script.")