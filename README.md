# AMD_Robotics_Hackathon_2025_Automated_Printing

## Team Information

**Team:** *Rainy Night*

**Summary:** *A brief description of your work*


---

# AMD Robotic Hackathon 2025 — Automated Printing

### **Team: Rainy Nights**

- Shunsuke Kanzawa

- Taro Sukegawa

- Ryudai Yokokawa

- Takeru Igarashi

---

## 🎬 Demo

```markdown
<video src="resources/20251207_amdhackathon.mp4" controls width="600"></video>
```

---

## 概要 / Summary

## 🧭 1. ミッション

本プロジェクトは、**オープンソースかつ低コストなロボットアーム SO-101 を活用し、3Dプリンタの「後処理」を自動化する**ことを目指しています。
近年、3Dプリンタは個人から企業まで幅広く普及し、遠隔印刷や自動キューイングなど「印刷の自動化」は進んでいます。しかし、**印刷後の取り出し・片付けは依然として人力に依存**しており、ユーザーはプリンタに縛られ続けています。

本システムは、SO-101 の汎用性を最大限に活かし、**既存の高額なロボットソリューションが不要な “現実的な自動化”** を実現します。これにより、ユーザーは3Dプリンタから解放され、時間をより自由に使える未来を目指します。

---

## 🎨 2. Creativity / 創造性

私たちは、**3Dプリンタ利用者が抱える「待ち時間と後処理の負担」を模倣学習で解決する**アプローチを採用しました。

* **機能ごとに学習データを分割**し、誰でも編集しやすい構成
* 汎用性が求められる動作（取り出し・物体操作）は **模倣学習（IL）** により柔軟性と再現性を確保
* 一方、単純なタスク（清掃など）は **Rl-reply ベースの動作**として実装し、ユーザーが環境に合わせて簡単に調整可能

これにより、システムとしてのシンプルさを保ったまま、**「誰もが求める自動化」**を実現しています。

---

## 🛠️ 3. 技術的な実装

* **使用プリンタ**：Bambu Lab A1 mini
* **ロボットアーム**：SO-101（1本）
* **プリントモデル数**：13種＋意図的に用意した失敗ワーク
* **データセット**：110エピソード
  * 85エピソード：ワークの移動
  * 25エピソード：スパゲッティ（失敗出力）の処理
* **バックエンド**：HuggingFace *lerobot*

---

## 👍 4. Ease of Use / 使いやすさ

本システムは専門知識や高価な設備を必要とせず、**誰でも導入できる“実用的なロボット後処理システム”**として設計しています。

### ✔ 再現性：オープンソース × 安価なハードウェア

* ソースコードは完全オープンソース
* ハードウェアには **約 300 USD の SO-101** を採用
* 低コストで産業用ロボット並みの自動化を再現可能

### ✔ 高い汎用性：多くの3Dプリンタに後付け可能

* 特定のプリンタに依存しない設計
* Web UI によるオンライン制御
* トップカメラの設置スペースとアーム可動域が確保できれば、多くの市販プリンタに適用可能

### ✔ 多様な物体に対応できる認識モデル

白黒5種のブロック、スクレーパー、スマホスタンドなど、形状・コントラストの異なる多様な物体で学習。
学習に含まれていない **船型の複雑モデルも把持に成功**し、汎化性能の高さを確認。

---

## 🚀 拡張性

本システムは単なる「取り出し装置」ではなく、**多目的ロボットプラットフォーム**として拡張可能です。

### 🔹 未知物体への対応

* 未学習の形状でも把持位置を推論
* モデルを変えるたびに再学習不要
* シームレスな連続運用が可能

### 🔹 多目的タスクの統合

* スクレーパーによるベッド清掃
* 取り出した物体の分類・整列（テトリス配置など）
* ソフト側のタスクフローを変えるだけで新タスクを追加可能

### 🔹 ハードウェア非依存の設計

物体認識とアーム制御を独立モジュール化。
SO-101 以外のアームや別カメラへの移行も最小限の変更で対応可能。
規模の大きい製造プロセスへの応用も視野に入ります。

---

## 📦 Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-record--3dprint--clean-blue)](https://huggingface.co/datasets/RainyNight17/record-3dprint-clean)

[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-record--calib--clean-blue)](https://huggingface.co/datasets/RainyNight17/record-calib-clean)


## 🤖 Model

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Policy--3DPrint--Clean--Test-blue)](https://huggingface.co/RainyNight17/policy-3dprint-clean-test/tree/main)
---