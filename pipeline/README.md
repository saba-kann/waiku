# 自動採譜パイプライン

「96%の自動採譜手順」を、失われた手順そのものを推測で再現するのではなく、
**この環境で検証できる正確な情報だけ**(BPM・キー、ユーザー確認済みの正本メロディ、
KeyTubeで裏付けられた構造情報)を使って再構築したもの。詳細な経緯は `CLAUDE.md` 参照。

## できること・できないこと

- **できる(このリポジトリの環境で実行可): ピッチ追跡・浄化・ノーツ化・評価・オフボーカル差分**
  (`pitch_track.py` / `informed_separate.py` / `notes.py` / `evaluate.py` / `subtract.py`)。
  ネットワーク接続不要
- **できない: Demucsによるボーカル分離** (`separate.py`)。モデル重みのダウンロード先
  (huggingface.co)がこの環境のネットワークポリシーで403ブロックされている
  (2026-09-27確認)。分離だけは重みを取得できる環境(手元PC等)で行い、
  出力された `vocals.wav` をこの環境にアップロードすること

**オフボーカル(インスト)音源が手に入るなら、Demucsより先に`subtract.py`を試すこと。**
Demucsの機械学習による推定と違い、原曲からオフボーカルを厳密に引き算するだけなので、
時刻・音量さえ合わせられればノイズが少ない。2026-09-27の実測ではDemucsなしでも
frame一致率70%前後まで出た(詳細はCLAUDE.md参照)。ただしこの曲でオフボーカル音源が
手に入ったのは偶然で、どの曲にも使える保証はない(最終ゴールの一般化にはDemucs系の
パスも要る)。

## 使い方

```bash
# 1a. オフボーカル音源があるなら(こちらを優先): 差分でボーカルを取り出す
python3 pipeline/subtract.py 原曲.mp3 オフボーカル.mp3 vocal_isolated.wav

# 1b. なければ(別環境で) Demucsで分離
python3 pipeline/separate.py 原曲.mp3   # → separated/htdemucs_ft/原曲/vocals.wav

# 2. 正本がある区間で精度を測る(パラメータ調整・信頼性の確認に使う)
python3 pipeline/transcribe.py calibrate vocal_isolated_0_132.wav data/vocal_melody_0_132s.json \
    --repeat-block 88:106

# 3. 未知区間を書き起こす(自己参照で反復浄化)
python3 pipeline/transcribe.py transcribe vocal_isolated_165_241.wav --offset 165 \
    --out candidates_165-241_auto.json
```

`calibrate`は「浄化なしの生pYIN」と「正本を使ったscore-informed浄化」の両方の精度を
出す。後者が前者よりはっきり良ければ、浄化のロジック自体は機能している証拠になる。

`--repeat-block start:end` は、歌詞の反復句など同じ高さの音が近接して並ぶ区間を指定する。
指定すると、その区間内では
  - 無声の谷を安易にまたいで1つの音にまとめない
  - 同じ高さの隣接ノートを再マージしない
  - オンセット検出の結果を使って、無声区間なしの再アタックも音符境界として拾う
という扱いになる。**指定しないと88〜106秒のような区間が丸ごと1音に潰れるバグ
(過去に実際に起きた)を再現してしまうので注意。**

## ファイル

- `pitch_track.py` ― pYINでのピッチ追跡(`pyin_track`/`pyin_track_array`)とオンセット検出
  (`detect_onsets`)。ネットワーク不要
- `informed_separate.py` ― 期待ピッチ曲線を使ったハーモニックマスク浄化
  (`clean_with_prior`)。ピアノ漏れなど、声の倍音上に乗らない周波数成分を減衰させる
- `notes.py` ― 連続ピッチ曲線を離散ノーツに変換(`segment_notes`)。repeat_block対応
- `evaluate.py` ― 正本との突き合わせ(フレーム単位の一致率、ノート単位のprecision/recall/F1)
- `separate.py` ― Demucsラッパー(この環境では実行不可。上記参照)
- `subtract.py` ― オフボーカル音源との差分でボーカルを取り出す(`subtract_vocal`/
  `subtract_vocal_files`)。時刻ズレは相互相関で自動検出、音量差は最小二乗でスケール推定
- `transcribe.py` ― 上記をつなぐCLI(`calibrate`/`transcribe`サブコマンド)
- `selftest.py` ― 合成音声での回帰テスト。`python3 pipeline/selftest.py`で実行。
  実装中に実際に2つのバグ(repeat_block内の谷またぎ判定・マージ判定の位置ズレ)を
  この場で検出・修正した

## 既知の限界(合成音声でのテストから分かったこと)

- オンセット検出は完璧ではなく、repeat_block内でやや過剰に分割する傾向がある
  (合成テストで8音の反復句が9〜10音に分かれた)。実音声でパラメータ調整が必要
- `informed_separate.clean_with_prior`はSTFTベースのハーモニックマスクで、
  倍音構造が薄い子音部や息継ぎ音には効きにくい可能性がある。実音声で要検証
- `semitone_jump`(音符境界とみなすピッチジャンプの閾値、既定0.6半音)や
  `min_note_dur`(既定0.03秒)は未調整。`calibrate`で正本と突き合わせてから
  値を決めること
