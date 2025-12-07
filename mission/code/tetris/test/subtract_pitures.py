import cv2
import numpy as np
import os

def crop_image(img, x, y, w, h):
    """指定範囲をクロップ"""
    return img[y:y+h, x:x+w]

def compare_images(image1_path, image2_path, threshold_value=50, crop_area=None):
    # 画像を読み込む
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("画像が読み込めませんでした。パスを確認してください。")
        return

    # クロップ指定がある場合
    if crop_area:
        x, y, w, h = crop_area
        img1 = crop_image(img1, x, y, w, h)
        img2 = crop_image(img2, x, y, w, h)

    # サイズが異なる場合は揃える
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 差分計算
    diff = cv2.absdiff(img1, img2)

    # グレースケール化
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 閾値処理
    _, thresh = cv2.threshold(gray_diff, threshold_value, 255, cv2.THRESH_BINARY)

    # 保存
    path_threshold = f"threshold_{threshold_value}.jpg"
    cv2.imwrite("crop1.jpg", img1)
    cv2.imwrite("crop2.jpg", img2)
    cv2.imwrite("difference.jpg", diff)
    cv2.imwrite(path_threshold, thresh)

    print("保存しました:")
    print(" - crop1.jpg")
    print(" - crop2.jpg")
    print(" - difference.jpg")
    print(f" - threshold_{threshold_value}.jpg")

    return thresh

def clustering(path_threshold):
    img = cv2.imread(path_threshold)
    data = img.reshape((-1, 3))
    data = np.float32(data)

    K = 2  # 3クラスに分ける
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    clustered = centers[labels.flatten()]
    clustered = clustered.reshape(img.shape)

    cv2.imwrite("clustered.jpg", clustered)    

def remove_small_white_regions(path_threshold):
    # 白黒画像として読み込み
    img = cv2.imread(path_threshold, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERROR: 画像が読み込めません:", path_threshold)
        return

    # 輪郭検出（白=255 を対象）
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("白領域なし")
        return

    # 面積で最大の白領域を探す
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_index = np.argmax(areas)
    max_contour = contours[max_index]

    # 新しい画像（黒）を作成
    filtered = np.zeros_like(img)

    # 最大の白領域だけ描画
    cv2.drawContours(filtered, [max_contour], -1, 255, thickness=-1)

    cv2.imwrite("largest_region.jpg", filtered)
    print("最大白領域を抽出 → largest_region.jpg として保存しました")

    return filtered

def blob_detection(binary_img, blob_thrh=100):
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

    print(f"検出された輪郭の総数: {len(contours)}")
    detected_count = 0

    for contour in contours:
        # 輪郭の面積を計算
        area = cv2.contourArea(contour)
        
        # 面積が閾値より大きい輪郭のみをテトロミノとして処理
        if area > blob_thrh:
            detected_count += 1
            
            # --- 最小外接矩形 (minAreaRect) の計算と描画 ---
            
            # 1. minAreaRect を計算
            rect = cv2.minAreaRect(contour)
            
            # 2. 矩形の4隅の頂点座標を計算 (boxPoints)
            # rect の情報を基に、矩形の4つの頂点座標を計算
            box = cv2.boxPoints(rect)
            
            # 3. 座標を整数型に変換（描画用）
            box = box.astype(int)
            
            # 4. 矩形の描画 (drawContours)
            # result_img: 描画対象の画像
            # [box]: 描画する頂点座標のリスト
            # 0: 描画する輪郭のインデックス
            # (255, 0, 0): 色 (例: 青色)
            # 2: 線幅
            cv2.drawContours(result_img, [box], 0, (255, 0, 0), 2)
            
            # --- 従来の boundingRect の描画（残す場合） ---
            
            # 軸に沿った矩形も描画したい場合は、この行を残す
            # x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 1) # 緑色、細線に変更    print(f"フィルタリング後に検出されたテトロミノの数: {detected_count}")

    # 5.1. 画像の書き出し (imwrite)
    output_filename = 'tetromino_detected.png'
    cv2.imwrite(output_filename, result_img)
    print(f"結果画像を {output_filename} に書き出しました。")


if __name__ == "__main__":
    image1_path = "other_files/pictures_objects/cube.jpeg"
    image2_path = "other_files/pictures_objects/none.jpeg"

    # 例：クロップ範囲を指定（x, y, w, h）
    crop_area = (250, 200, 220, 220)

    threshold_value = int(100)

    thresh = compare_images(image1_path, image2_path, threshold_value, crop_area)
    #path_filtered = remove_small_white_regions(path_threshold)
    blob_detection(thresh)
