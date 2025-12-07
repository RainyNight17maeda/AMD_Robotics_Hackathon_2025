from lerobot.common.robot_devices.so100 import SO100FollowerRobot
from lerobot.common.policies.base_policy import Policy
from huggingface_hub import login
import time

# --- (1) HF login（必要なら）
# login("hf_xxxxxxxxxxxxxxxxxxxxx")

# --- (2) Robot の初期化（Ubuntu のコマンドを Python 化）
robot = SO100FollowerRobot(
    port="/dev/ttyACM1",
    cameras={
        "up": {"type": "opencv", "index_or_path": "/dev/video10", "width": 640, "height": 480, "fps": 30},
        "side": {"type": "intelrealsense", "serial_number_or_name": "233522074606", "width": 640, "height": 480, "fps": 30},
    },
    robot_id="my_awesome_follower_arm"
)

# --- (3) Hugging Face Hub 上のモデルからポリシーを読み込む
policy = Policy.from_pretrained("HF_USER/my_policy")  # ★ダウンロード不要（自動キャッシュ）

# --- (4) 推論 5 回分のループ
print("Start inference...")
for i in range(5):
    # ロボットから観測データ取得
    obs = robot.get_observation()

    # ポリシー推論（アクション生成）
    action = policy.predict(obs)

    # ロボットへアクション送信
    robot.apply_action(action)

    print(f"[{i+1}/5] step done")
    time.sleep(0.05)  # 20Hz 例

print("Inference finished.")

# --- (5) ロボットを安全に停止
robot.stop()
