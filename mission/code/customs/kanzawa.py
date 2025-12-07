from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from huggingface_hub import login

from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.scripts.lerobot_record import RecordConfig


login("token")
NUM_EPISODES = 5
FPS = 15
EPISODE_TIME_SEC = 25
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "RainyNight17/policy-vscode-test"
HF_DATASET_ID = "RainyNight17/eval_vscode-test"

# Create the robot configuration
camera_config = {"front": OpenCVCameraConfig(index_or_path="/dev/video6", width=640, height=480, fps=FPS),"top": OpenCVCameraConfig(index_or_path="/dev/video5", width=640, height=480, fps=FPS)}

#  Follower (SO101/SO100)
follower_cfg = SO101FollowerConfig(
    port="/dev/ttyACM1", id="my_awesome_follower_arm", cameras=camera_config,
)
follower = SO101Follower(follower_cfg)

#  Leader (SO100 Leader teleop)
leader_cfg = SO101LeaderConfig(
    port="/dev/ttyACM0", id="my_awesome_leader_arm",
)
leader = SO101Leader(leader_cfg)

# Initialize the policy
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=follower.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot
follower.connect()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

dataset_features = combine_feature_dicts(
    aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(
            action=follower.action_features
        ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
        use_videos=dataset.video,
    ),
    aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=follower.observation_features),
        use_videos=dataset.video,
    ),
)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=follower,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
    

    dataset.save_episode()

# Clean up
follower.disconnect()
dataset.push_to_hub()