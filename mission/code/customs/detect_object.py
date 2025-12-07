import cv2
import time
import os

def capture_and_save_image(camera_index=0):
    # カメラを初期化
    cap = cv2.VideoCapture(camera_index)
    
    # カメラが正しく開けるか確認
    if not cap.isOpened():
        print("カメラが開けませんでした。")
        return
    
    # 1秒待機（カメラが安定するまで）
    time.sleep(1)
    
    # フレームをキャプチャ
    ret, frame = cap.read()
    
    if ret:
        while True:
            # ユーザーに保存する画像の名前を入力してもらう
            file_name = input("保存する画像の名前を入力してください（拡張子を含む、例: image.jpg）：")
            
            # ファイルがすでに存在するか確認
            if os.path.exists(file_name):
                print(f"エラー: '{file_name}' はすでに存在します。別の名前を入力してください。")
            else:
                # 画像を指定された名前で保存
                cv2.imwrite(file_name, frame)
                print(f"画像が保存されました: {file_name}")
                break
    else:
        print("画像のキャプチャに失敗しました。")
    
    # カメラを解放
    cap.release()

# 使用例：USBカメラから画像をキャプチャして保存
capture_and_save_image(camera_index=9)
