# ====================
#  Camera configuration
# ====================

camera_config = {
    "up": OpenCVCameraConfig(
        index_or_path="/dev/video10",
        width=640, height=480, fps=30
    ),
    "side": IntelRealSenseCameraConfig(
        serial_number_or_name="233522074606",
        width=640, height=480, fps=30
    )
}

# ====================
#  Follower (SO101/SO100)
# ====================

follower_cfg = SO100FollowerConfig(
    port="/dev/ttyACM1",
    id="my_awesome_follower_arm",
    cameras=camera_config,
)

follower = SO100Follower(follower_cfg)


# ====================
#  Leader (SO100 Leader teleop)
# ====================

leader_cfg = SO100LeaderConfig(
    port="/dev/ttyACM0",
    id="my_awesome_leader_arm",
)

leader = SO100Leader(leader_cfg)


# ====================
#  Dataset creation
# ====================

# follower から feature を抽出
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")

dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=f"{HF_USER}/eval_so100",
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
    fps=30,
    episode_time_s=60,
)

dataset.save_episode()

# ====================
#  Cleanup
# ====================

follower.disconnect()
leader.disconnect()

dataset.push_to_hub()
