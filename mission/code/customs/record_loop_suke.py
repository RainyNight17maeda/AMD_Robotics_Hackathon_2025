import time
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

# ====================
#  Username + directory
# ====================
HF_USER = 'RainyNight17'
REPO_ID = f"{HF_USER}/eval_so110"

# ====================
#  Camera configuration
# ====================
camera_config = {
    "up": OpenCVCameraConfig(
        index_or_path="/dev/video5",
        width=640, height=480, fps=15
    ),
    "side": OpenCVCameraConfig(
        index_or_path="/dev/video6",
        width=640, height=480, fps=15
    )
}

# ====================
#  Other properties
# ====================
EPISODE_TIME_S = 60

def manual_record_loop(
    follower: SO101Follower,
    leader: SO101Leader,
    dataset: LeRobotDataset,
    single_task: str,
    fps: int,
    episode_time_s: int,
):
    dt = 1.0 / fps
    start = time.time()
    while time.time() - start < episode_time_s:
        # 1. Leader から現在の動作（action）を取得
        action = leader.get_action()   # ← teleop の基本API
        # 2. follower に action を送信
        follower.send_action(action)
        #follower.apply_action(action)
        # 3. 観測を取得
        obs = follower.get_observation()
        # 4. dataset に {obs, action} を追加
        frame = {
            "observation":obs,
            "action":action,
            "task": single_task,

        }
        dataset.add_frame(frame)
        #dataset.append_step(
        #    observation=obs,
        #    action=action,
        #)
        time.sleep(dt)

# ====================
#  Follower (SO101/SO100)
# ====================
follower_cfg = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="my_awesome_follower_arm",
    cameras=camera_config,
)
follower = SO101Follower(follower_cfg)

# ====================
#  Leader (SO100 Leader teleop)
# ====================
leader_cfg = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="my_awesome_leader_arm",
)
leader = SO101Leader(leader_cfg)

# ====================
#  Dataset creation
# ====================
# follower から feature を抽出
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}
dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=30,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
)

# ====================
#  Connect devices
# ====================
leader.connect()
follower.connect()

# ====================
#  Run one episode
# ====================
manual_record_loop(
    follower=follower,
    leader=leader,
    dataset=dataset,
    fps=15,
    episode_time_s=EPISODE_TIME_S,
    single_task="test"
)

dataset.save_episode()

# ====================
#  Cleanup
# ====================

follower.disconnect()
leader.disconnect()

dataset.push_to_hub()

