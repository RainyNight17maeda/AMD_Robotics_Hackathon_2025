#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom version of `lerobot-record` for LeRobot v0.4.x.

- Uses the same CLI / config style as the official `lerobot_record.py`
- Adds OpenCV-based preprocessing on a "top" camera image
- Stores the results as:
    observation.custom.top_points  (float32, shape = (MAX_POINTS, 2))
    observation.custom.top_scores  (float32, shape = (MAX_POINTS,))
so they can be used both for dataset creation and for policy inference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import rerun as rr

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # exported for the config system
)
from lerobot.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.cameras.realsense.configuration_realsense import (  # noqa: F401
    RealSenseCameraConfig,
)
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    build_dataset_frame,
    hw_to_dataset_features,
)
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    koch_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# ==========================
# Custom feature definitions
# ==========================

# top カメラ画像が observation に入ってくるキー
# 例: --robot.cameras="{ top: {type: opencv, index_or_path: 0, ...}}"
#     → observation["images.top"] に画像が入ることが多いです。
TOP_CAMERA_HW_KEY = "images.top"

# ここで追加する独自観測のキー（observation dict 内のキー）
CUSTOM_POINTS_HW_KEY = "custom.top_points"
CUSTOM_SCORES_HW_KEY = "custom.top_scores"

# データセット上の最大点数（固定長にする必要がある）
CUSTOM_MAX_POINTS = 64


def _compute_points_from_top_image(
    img: np.ndarray,
    max_points: int = CUSTOM_MAX_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    任意の前処理を書く場所。
    ここでは簡単な例として:
    - グレースケール＋二値化
    - 輪郭抽出
    - 各輪郭の重心 (cx, cy) を座標として使う
    - 面積をスコアとして使う
    """

    # img は H x W x 3 の np.ndarray（BGR or RGB）を想定
    if img.ndim != 3 or img.shape[2] != 3:
        # 想定外の形状なら空配列を返す
        points = np.zeros((max_points, 2), dtype=np.float32)
        scores = np.zeros((max_points,), dtype=np.float32)
        return points, scores

    # OpenCV 側では BGR 前提なので、とりあえずそのまま扱う
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 閾値は適宜調整（ここでは明るい部分を「ゴミっぽい」と仮定）
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # ノイズ除去など必要ならここで追加（膨張・収縮 etc.）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pts: list[Tuple[float, float]] = []
    scs: list[float] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        pts.append((cx, cy))
        scs.append(float(area))

    # スコアの大きい順にソート
    if len(pts) > 0:
        order = np.argsort(np.array(scs))[::-1]
        pts = [pts[i] for i in order]
        scs = [scs[i] for i in order]

    # 固定長にパディング／トリム
    num = min(len(pts), max_points)
    points = np.zeros((max_points, 2), dtype=np.float32)
    scores = np.zeros((max_points,), dtype=np.float32)

    if num > 0:
        points[:num, :] = np.asarray(pts[:num], dtype=np.float32)
        scores[:num] = np.asarray(scs[:num], dtype=np.float32)

    return points, scores


def add_custom_observation_fields(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    robot.get_observation() で得られた辞書に
    CUSTOM_POINTS_HW_KEY / CUSTOM_SCORES_HW_KEY を追加する。
    """

    img = obs.get(TOP_CAMERA_HW_KEY, None)
    if img is None:
        # top カメラが無い場合は何もせずそのまま返す
        return obs

    points, scores = _compute_points_from_top_image(img)

    obs[CUSTOM_POINTS_HW_KEY] = points
    obs[CUSTOM_SCORES_HW_KEY] = scores

    return obs


def get_custom_dataset_features() -> Dict[str, Dict[str, Any]]:
    """
    LeRobot の features 形式で、追加したい 2 つの観測を定義する。
    hw_to_dataset_features の出力と同じフォーマットに揃える。
    """
    feats: Dict[str, Dict[str, Any]] = {}

    # dataset.features のキーは "observation.xxx" 形式
    points_key = f"observation.{CUSTOM_POINTS_HW_KEY}"
    scores_key = f"observation.{CUSTOM_SCORES_HW_KEY}"

    feats[points_key] = {
        "dtype": "float32",
        "shape": (CUSTOM_MAX_POINTS, 2),
        "names": ["point", "xy"],
    }
    feats[scores_key] = {
        "dtype": "float32",
        "shape": (CUSTOM_MAX_POINTS,),
        "names": ["point"],
    }
    return feats


# =======================
# Config dataclasses (v0.4
# =======================


@dataclass
class DatasetRecordConfig:
    # Dataset identifier (e.g. "yourname/so101_garbage_pickup")
    repo_id: str

    # Description of the task, used in dataset metadata
    single_task: str

    # Where to store the dataset locally (default: HF cache directory)
    root: str | Path | None = None

    # Recording FPS
    fps: int = 30

    # Seconds per episode
    episode_time_s: float = 60.0

    # Seconds for reset between episodes
    reset_time_s: float = 60.0

    # Number of episodes to record
    num_episodes: int = 50

    # Save images as videos
    video: bool = True

    # Push dataset to the Hub after recording
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None

    # Image writer parallelism
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4

    def __post_init__(self) -> None:
        if self.single_task is None:
            raise ValueError("You must provide `dataset.single_task`.")


@dataclass
class RecordConfig:
    # Robot + dataset configs are parsed automatically by lerobot.configs.parser
    robot: RobotConfig
    dataset: DatasetRecordConfig

    # Either teleoperation, or a policy, or both must be provided
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None

    # Visualization / misc
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self) -> None:
        # Hack used by LeRobot to support `--policy.path=...`
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path,
                cli_overrides=cli_overrides,
            )
            self.policy.pretrained_path = policy_path  # type: ignore[attr-defined]

        if self.teleop is None and self.policy is None:
            raise ValueError(
                "You must choose at least one of: teleoperator or policy."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        # Enables `--policy.path` to be loaded through the parser.
        return ["policy"]


# =======================
# Main record loop
# =======================


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: Dict[str, Any],
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
) -> None:
    """
    Core loop used both for:
      - recording episodes
      - reset phases between episodes
    """

    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"Dataset fps must match requested fps ({dataset.fps} != {fps})."
        )

    if policy is not None:
        policy.reset()

    timestamp = 0.0
    start_episode_t = time.perf_counter()

    # control_time_s は呼び出し側で必ずセットされる想定
    assert control_time_s is not None

    while timestamp < control_time_s:
        loop_start = time.perf_counter()

        if events.get("exit_early", False):
            events["exit_early"] = False
            break

        # 1) Get observation from robot
        observation: Dict[str, Any] = robot.get_observation()

        # 2) Add custom fields computed from the top camera
        observation = add_custom_observation_fields(observation)

        # 3) Build dataset-style observation frame (if needed)
        if policy is not None or dataset is not None:
            # dataset.features は record() 内で構築している
            obs_frame = build_dataset_frame(
                dataset.features,  # type: ignore[arg-type]
                observation,
                prefix="observation",
            )
        else:
            obs_frame = None

        # 4) Decide action (policy or teleop)
        if policy is not None and obs_frame is not None:
            action_values = predict_action(
                obs_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {
                key: float(action_values[i].item())
                for i, key in enumerate(robot.action_features)
            }
        elif policy is None and teleop is not None:
            action = teleop.get_action()
        else:
            logging.info(
                "No policy / teleop → skipping action. "
                "This typically happens during reset without teleop."
            )
            # Wait for next tick but don't send actions / record
            dt = time.perf_counter() - loop_start
            busy_wait(max(0.0, 1.0 / fps - dt))
            timestamp = time.perf_counter() - start_episode_t
            continue

        # 5) Send action to robot
        sent_action = robot.send_action(action)

        # 6) Save to dataset
        if dataset is not None and obs_frame is not None:
            act_frame = build_dataset_frame(
                dataset.features,
                sent_action,
                prefix="action",
            )
            frame = {**obs_frame, **act_frame}
            dataset.add_frame(frame, task=single_task)

        # 7) Optional visualization via rerun
        if display_data:
            for obs_name, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs_name}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    # For images, rr.Image expects HWC
                    rr.log(f"observation.{obs_name}", rr.Image(val), static=True)
            for act_name, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act_name}", rr.Scalar(val))

        # 8) FPS control
        dt = time.perf_counter() - loop_start
        busy_wait(max(0.0, 1.0 / fps - dt))
        timestamp = time.perf_counter() - start_episode_t


# =======================
# Top-level entry point
# =======================


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    """
    Entry point used by `python lerobot_record_custom.py` (or as module).
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        _init_rerun(session_name="recording")

    # Build robot & teleop
    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop)
        if cfg.teleop is not None
        else None
    )

    # Base dataset features from robot hardware
    action_features = hw_to_dataset_features(
        robot.action_features,
        "action",
        cfg.dataset.video,
    )
    obs_features = hw_to_dataset_features(
        robot.observation_features,
        "observation",
        cfg.dataset.video,
    )

    # Our custom extra observation features
    custom_features = get_custom_dataset_features()

    dataset_features: Dict[str, Dict[str, Any]] = {
        **action_features,
        **obs_features,
        **custom_features,
    }

    # Create or resume dataset
    if cfg.resume:
        # 既存データセットに追加で録画したい場合
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras),
            )

        # NOTE: 既存データセットが custom_features を持っていないと
        #       ここでエラーになる可能性があります。
        sanity_check_dataset_robot_compatibility(
            dataset,
            robot,
            cfg.dataset.fps,
            dataset_features,
        )
    else:
        # 新規作成
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=(
                cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras)
                if hasattr(robot, "cameras")
                else 0
            ),
        )

    # Load pretrained policy if requested
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # Connect devices
    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    # Main episode loop
    for ep in range(cfg.dataset.num_episodes):
        log_say(
            f"Recording episode {ep + 1} / {cfg.dataset.num_episodes}",
            cfg.play_sounds,
        )

        # 1) Normal recording phase
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop=teleop,
            policy=policy,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )

        # 2) Optional reset phase (skip after last episode)
        if (
            not events.get("stop_recording", False)
            and (
                ep < cfg.dataset.num_episodes - 1
                or events.get("rerecord_episode", False)
            )
        ):
            log_say("Reset the environment", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=teleop,
                policy=None,  # reset フェーズは policy なしが多い
                dataset=None,  # データは保存しない
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

        # Re-record logic
        if events.get("rerecord_episode", False):
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # 保存
        dataset.save_episode()

        if events.get("stop_recording", False):
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    # Clean up
    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(
            tags=cfg.dataset.tags,
            private=cfg.dataset.private,
        )

    log_say("Exiting", cfg.play_sounds)

    return dataset


if __name__ == "__main__":
    # `python lerobot_record_custom.py --robot.type=so101_follower ...`
    # のように実行できる
    record()

	
