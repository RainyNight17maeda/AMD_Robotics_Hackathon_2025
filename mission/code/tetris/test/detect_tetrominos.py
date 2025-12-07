import cv2
import numpy as np
import math
import os

debug_image_dir = None #"project/images"

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
    save_image("crop1.jpg", img1)
    save_image("crop2.jpg", img2)
    save_image("difference.jpg", diff)
    save_image(path_threshold, thresh)

    return thresh

def save_image(name, img):
    if debug_image_dir:
        cv2.imwrite(os.path.join(debug_image_dir, name), img)
        print(f"debug image saved: {name}")  

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

    detected_count = 0

    for contour in contours:
        # 輪郭の面積を計算
        area = cv2.contourArea(contour)
        
        # 面積が閾値より大きい輪郭のみをテトロミノとして処理
        if area > blob_thrh:
            detected_count += 1
            # 1. minAreaRect を計算
            rect = cv2.minAreaRect(contour)
            
            shape = identify_tetro_shape(binary_img, rect)

            if debug_image_dir:
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(result_img, [box], 0, (255, 0, 0), 2)
                cx, cy = rect[0]
                cv2.putText(result_img, shape, (int(cx), int(cy)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))

    # 5.1. 画像の書き出し (imwrite)
    save_image('detected_rects.png', result_img)


def identify_tetro_shape(binary_img, rect):
    """
    テトロミノの輪郭とminAreaRectの結果から、I, O, T, L, J, S, Zの
    いずれかの形状を識別する関数。

    Args:
        binary_img (np.ndarray): テトロミノを含む元の二値化画像 (0: 黒, 255: 白)。
        rect (tuple): cv2.minAreaRectの出力 (中心, サイズ(w, h), 角度)。

    Returns:
        str: 識別された形状名 ("T", "L", "J", "S", "Z", "I", "O")。
    """
    
    (center_x, center_y), (width, height), angle = rect
    
    # 長辺と短辺を決定
    long_side = max(width, height)
    short_side = min(width, height)

    epsilon = 1e-6
    aspect_ratio = long_side / (short_side + epsilon)
    
    # 長方形の縦横比の閾値 (I形、O形の識別)
    I_SHAPE_THRESHOLD = 2.5  # 4.0 に近い値
    O_SHAPE_THRESHOLD = 1.2  # 1.0 に近い値
    
    if aspect_ratio >= I_SHAPE_THRESHOLD:
        # 非常に細長い場合、I形と識別
        return "I"
    
    if aspect_ratio <= O_SHAPE_THRESHOLD:
        # ほぼ正方形の場合、O形と識別
        return "O"
    
    # セルサイズを推定 (短辺が2セル分、長辺が3セル分と仮定)
    # これにより、サンプリング時の座標を正確に計算できる
    cell_size = short_side / 2.0
    
    # 識別用に画像を回転補正して切り抜く (2x3の正規化された向きにする)
    
    # 矩形の幅と高さの順序を、rotated_rectのサイズと同じ順序にする
    if width < height:
        # width < height (縦長) の場合、90度回転させる
        M = cv2.getRotationMatrix2D((center_x, center_y), angle + 90, 1.0)
        output_width, output_height = math.ceil(height), math.ceil(width)
    else:
        # width > height (横長) の場合、そのままの角度
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        output_width, output_height = math.ceil(width), math.ceil(height)
    
    # 座標を移動し、切り抜き後の画像が中心に来るようにする
    M[0, 2] += output_width / 2.0 - center_x
    M[1, 2] += output_height / 2.0 - center_y
    
    # 回転補正と切り抜き
    # 外接長方形のサイズに合わせて切り抜くが、境界線上のノイズを避けるため、
    # わずかに大きめに領域を取る (padding)
    padding = 0
    final_w = int(output_width + padding * 2)
    final_h = int(output_height + padding * 2)
    
    # Mを更新してpaddingを考慮した移動
    M[0, 2] += padding
    M[1, 2] += padding
    
    rotated_cropped_img = cv2.warpAffine(binary_img, M, (final_w, final_h), 
                                         flags=cv2.INTER_NEAREST, borderValue=255)
    save_image("rot_clip.png", rotated_cropped_img)
    # ------------------------------------------------
    # 識別ロジック: 2x3 グリッドでのピクセルサンプリング
    # ------------------------------------------------
    
    # サンプリング点の定義 (切り抜き画像 (final_w x final_h) の中心からの相対座標)
    # ここでは、短辺方向 (H) が2セル、長辺方向 (W) が3セルになるように正規化する。
    
    # グリッドのセルの中央間の距離
    center_dist_w = long_side / 3.0
    center_dist_h = short_side / 2.0
    
    # サンプリングのオフセット (中心 (final_w/2, final_h/2) から)
    # W方向 (3点): -1.0, 0.0, +1.0 * center_dist_w
    # H方向 (2点): -0.5, +0.5 * center_dist_h
    
    offsets_w = [-center_dist_w, 0, center_dist_w]
    offsets_h = [-center_dist_h / 2.0, center_dist_h / 2.0]
    
    center = (final_w // 2, final_h // 2)
    sample_radius = max(2, int(cell_size * 0.2)) # セルサイズの20%程度の円
    
    # ピクセルパターンを格納 (0: 黒/埋まっている, 1: 白/欠けている)
    # [h=0, w=0], [h=0, w=1], [h=0, w=2]
    # [h=1, w=0], [h=1, w=1], [h=1, w=2]
    pixel_pattern = np.zeros((2, 3), dtype=int)
    
    # サンプリング実行
    for h_idx, dy in enumerate(offsets_h):
        for w_idx, dx in enumerate(offsets_w):
            
            # グローバル座標 (切り抜き画像内)
            sample_x = int(center[0] + dx)
            sample_y = int(center[1] + dy)
            
            # 円形マスクを作成
            mask = np.zeros((final_h, final_w), dtype=np.uint8)
            cv2.circle(mask, (sample_x, sample_y), sample_radius, 255, -1)
            
            # マスク領域の平均ピクセル値を取得
            mean_val = cv2.mean(rotated_cropped_img, mask=mask)[0]
            
            # 平均値が白いピクセル (255) に近ければ「欠けている (1)」、
            # 黒いピクセル (0) に近ければ「埋まっている (0)」と判定
            # 閾値は 255 / 2 = 127.5
            pixel_pattern[h_idx, w_idx] = 1 if mean_val > 127 else 0
            
    # パターンマッチングによる識別
    pattern_flat = tuple(pixel_pattern.flatten())
    patterns = {
        # T形
        (1, 1, 1, 0, 1, 0): "T", 
        (0, 1, 0, 1, 1, 1): "T", 
        # L形
        (1, 1, 1, 1, 0, 0): "L",
        (0, 0, 1, 1, 1, 1): "L",
        # J形 (L形の反転): 
        (1, 1, 1, 0, 0, 1): "J",
        (1, 0, 0, 1, 1, 1): "J",        
        # S形:
        (0, 1, 1, 1, 1, 0): "S",
        # Z形 (S形の反転):
        (1, 1, 0, 0, 1, 1): "Z",
        # I形 (IかOが混ざったときのフォールバック)
        (1, 1, 1, 1, 1, 1): "I",
    }
    
    return patterns.get(pattern_flat, None)

def detect_tetrominos(img, 
                      crop_area=(250, 200, 220, 220), 
                      bin_thrh=100, ):
    if __name__ == "__main__":
    image1_path = r"project\other_files\pictures_objects\all2.jpg"
    image2_path = r"project\other_files\pictures_objects\none2.jpg"

    # 例：クロップ範囲を指定（x, y, w, h）
    crop_area = (250, 200, 220, 220)

    threshold_value = int(100)

    thresh = compare_images(image1_path, image2_path, threshold_value, crop_area)
    #path_filtered = remove_small_white_regions(path_threshold)
    blob_detection(thresh)
    return None

if __name__ == "__main__":
    image1_path = r"project\other_files\pictures_objects\all2.jpg"
    image2_path = r"project\other_files\pictures_objects\none2.jpg"

    # 例：クロップ範囲を指定（x, y, w, h）
    crop_area = (250, 200, 220, 220)

    threshold_value = int(100)

    thresh = compare_images(image1_path, image2_path, threshold_value, crop_area)
    #path_filtered = remove_small_white_regions(path_threshold)
    blob_detection(thresh)
