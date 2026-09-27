"""Demucsでボーカルステムを取り出す薄いラッパー。

**重要: このリポジトリのClaude Code環境ではこの関数は動かない。**
モデル重みのダウンロード先(huggingface.co / dl.fbaipublicfiles.com)がこの環境の
ネットワークポリシーで403ブロックされている(2026-09-27に確認)。
CLAUDE.mdの「前環境ではDemucsの重みは手動ダウンロードが必要だった」という記述と一致する。

そのため分離ステップだけは、重みを取得できる環境(ユーザーの手元PC等)で
実行し、出力された vocals.wav をこちらにアップロードしてもらう運用にすること。
それ以降(pitch_track.py 以降)はこのリポジトリの環境だけで完結する。
"""
import subprocess
import sys
from pathlib import Path


def separate(input_audio, out_dir="separated", model="htdemucs_ft"):
    """`python3 -m demucs.separate --two-stems=vocals -n <model> -o <out_dir> <input_audio>` を実行し、
       vocals.wav のパスを返す。モデル重みのダウンロードが必要(上記の制約に注意)。"""
    cmd = [sys.executable, "-m", "demucs.separate", "--two-stems=vocals",
           "-n", model, "-o", out_dir, str(input_audio)]
    subprocess.run(cmd, check=True)
    stem = Path(out_dir) / model / Path(input_audio).stem / "vocals.wav"
    if not stem.exists():
        raise FileNotFoundError(f"expected output not found: {stem}")
    return str(stem)


if __name__ == "__main__":
    print(separate(sys.argv[1]))
