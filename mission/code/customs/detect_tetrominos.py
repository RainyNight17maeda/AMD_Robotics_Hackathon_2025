import cv2
import numpy as np


# ユーザー様が準備された二値画像を読み込む
# 例: binary_image.png は既に差分処理などでテトロミノの形状が抽出された二値画像
binary_img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)

# 画像が正しく読み込めたか確認
if binary_img is None:
    print("画像を読み込めませんでした。ファイルパスを確認してください。")
    exit()

# 二値画像であること（0と255のみ）を前提に処理を進めます。
# 必要に応じて、ここで追加のノイズ除去や閾値処理を行っても良いです。
# 例：
# _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

# cv2.findContours() を使用して輪郭を検出する
# cv2.RETR_EXTERNAL: 最も外側の輪郭のみを検出（テトロミノの各ピースの外周）
# cv2.CHAIN_APPROX_SIMPLE: 輪郭の冗長な点を削除し、必要な頂点のみを保持
contours, hierarchy = cv2.findContours(
    binary_img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 検出結果を描画するためのカラー画像を用意する（デバッグ・表示用）
# 元の二値画像と同じサイズ・型で3チャンネルの画像を生成し、黒で初期化
result_img = np.zeros((*binary_img.shape, 3), dtype=np.uint8)

# フィルタリングの閾値を設定 (ピクセル単位の面積)
# この値は画像サイズやテトロミノの大きさによって調整が必要です
MIN_AREA_THRESHOLD = 100 

print(f"検出された輪郭の総数: {len(contours)}")
detected_count = 0

for contour in contours:
    # 輪郭の面積を計算
    area = cv2.contourArea(contour)
    
    # 面積が閾値より大きい輪郭のみをテトロミノとして処理
    if area > MIN_AREA_THRESHOLD:
        detected_count += 1
        
        # 4.1. 外接矩形 (Bounding Rectangle) の計算
        # x, y: 矩形の左上隅の座標
        # w, h: 矩形の幅と高さ
        x, y, w, h = cv2.boundingRect(contour)

        # 1. 最小外接矩形の計算
        rect = cv2.minAreaRect(contour)

        # rect の情報を取り出す（オプション）
        (center_x, center_y), (width, height), angle = rect

        # 2. 矩形の4隅の頂点座標の計算
        # rect の情報を基に、矩形の4つの頂点座標を計算する
        box = cv2.boxPoints(rect)

        # 3. 座標を整数型に変換（描画用）
        box = np.int0(box)

        # 4. 矩形の描画
        # img: 描画対象の画像
        # [box]: 描画する頂点座標のリスト
        # (0, 0, 255): 色 (赤色)
        # 2: 線幅
        
        # 4.2. 画像に矩形を描画
        # result_img: 描画対象の画像
        # (x, y): 矩形の左上隅
        # (x + w, y + h): 矩形の右下隅
        # (0, 255, 0): 色 (緑色)
        # 2: 線幅
        #cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(result_img,[box],0,(0,0,255), 2)

        
        # 検出された輪郭自体も描画したい場合（オプション）
        # cv2.drawContours(result_img, [contour], -1, (255, 0, 0), 1) 

print(f"フィルタリング後に検出されたテトロミノの数: {detected_count}")

# 5.1. 画像の書き出し (imwrite)
output_filename = 'tetromino_detected.png'
cv2.imwrite(output_filename, result_img)
print(f"結果画像を {output_filename} に書き出しました。")


if __name__ == "__main__":
    image1_path = "other_files/pictures_objects/cube.jpeg"
    image2_path = "other_files/pictures_objects/none.jpeg"