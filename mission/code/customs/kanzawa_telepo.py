from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time


camera_config = {
    "front": OpenCVCameraConfig(index_or_path="/dev/video6", width=640, height=480, fps=15),"top": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=15)
}

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="my_awesome_follower_arm",
    cameras=camera_config
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="my_awesome_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)

    if hasattr(observation, "images") and "front" in observation.images:
        frame = observation.images["front"]

        clear_output(wait=True)
        plt.imshow(frame)
        plt.axis('off')
        display(plt.gcf())

    time.sleep(0.03) 