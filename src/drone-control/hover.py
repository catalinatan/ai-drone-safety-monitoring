import airsim
import time
import cv2
import numpy as np
import keyboard

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.hoverAsync().join()

print("Controls: w/a/s/d = move, z/x = up/down, q = quit")

vx = vy = vz = 0.0

try:
    while True:
        # ----- camera feed -----
        response = client.simGetImage("0", airsim.ImageType.Scene)
        if response:
            img1d = np.frombuffer(response, dtype=np.uint8)
            img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("Drone camera", img)
        cv2.waitKey(1)  # just to keep the window responsive

        # ----- keyboard control -----
        if keyboard.is_pressed('q'):
            break

        # forward/back
        if keyboard.is_pressed('w'):
            vx = 3
        elif keyboard.is_pressed('s'):
            vx = -3
        else:
            vx = 0

        # left/right
        if keyboard.is_pressed('d'):
            vy = 3
        elif keyboard.is_pressed('a'):
            vy = -3
        else:
            vy = 0

        # up/down (NED: negative z = up)
        if keyboard.is_pressed('z'):
            vz = -2
        elif keyboard.is_pressed('x'):
            vz = 2
        else:
            vz = 0

        client.moveByVelocityAsync(
            vx, vy, vz,
            duration=0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
        )

        time.sleep(0.03)

finally:
    cv2.destroyAllWindows()
    print("landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("done.")
