#!/usr/bin/env python3
"""自動採譜パイプラインの入口。分離済みボーカルwav(またはmix)から音符列を作る。

使い方:
  校正(正本がある区間で精度を測る):
    python3 pipeline/transcribe.py calibrate vocals_0_132.wav data/vocal_melody_0_132s.json

  未知区間の書き起こし(自己参照で反復浄化してからノーツ化):
    python3 pipeline/transcribe.py transcribe vocals_165_241.wav --offset 165
      --repeat-block 88:106 --out candidates_165-241_auto.json

separate.py(Demucs)はこの環境では実行できない(モデル重みのダウンロードがネットワーク
ポリシーでブロックされている)。ボーカルステムのwavは別環境で作ってアップロードすること。
"""
import argparse
import json
import sys

import numpy as np
import librosa

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pitch_track import pyin_track, pyin_track_array, detect_onsets
from informed_separate import clean_with_prior, notes_to_pitch_curve
from notes import segment_notes
from evaluate import frame_accuracy, note_level_prf


def parse_blocks(spec_list):
    out = []
    for spec in spec_list or []:
        a, b = spec.split(":")
        out.append((float(a), float(b)))
    return out


def cmd_calibrate(args):
    gold = [tuple(x) for x in json.load(open(args.gold))]
    y, sr = librosa.load(args.vocals, sr=None, mono=True)
    repeat_blocks = parse_blocks(args.repeat_block)
    onsets = detect_onsets(y, sr)

    times0, midi0, voiced0, _ = pyin_track_array(y, sr)
    naive = segment_notes(times0, midi0, voiced0, bpm=args.bpm,
                           repeat_blocks=repeat_blocks, onsets=onsets)
    naive_notes = [(n["start"] + args.offset, n["end"] + args.offset, n["midi"]) for n in naive]
    print("=== 浄化なし(生のpYIN) ===")
    print(frame_accuracy(naive_notes, gold))
    print(note_level_prf(naive_notes, gold))

    gold_curve_times = np.arange(0, times0[-1], 0.01)
    gold_shifted = [(s - args.offset, e - args.offset, p) for s, e, p in gold]
    gold_curve = notes_to_pitch_curve(gold_shifted, gold_curve_times)
    y_clean = clean_with_prior(y, sr, gold_curve_times, gold_curve)
    times1, midi1, voiced1, _ = pyin_track_array(y_clean, sr)
    cleaned = segment_notes(times1, midi1, voiced1, bpm=args.bpm,
                             repeat_blocks=repeat_blocks, onsets=onsets)
    cleaned_notes = [(n["start"] + args.offset, n["end"] + args.offset, n["midi"]) for n in cleaned]
    print("=== 正本で浄化(score-informed, 上限の目安) ===")
    print(frame_accuracy(cleaned_notes, gold))
    print(note_level_prf(cleaned_notes, gold))


def cmd_transcribe(args):
    y, sr = librosa.load(args.vocals, sr=None, mono=True)
    repeat_blocks = parse_blocks(args.repeat_block)
    onsets = detect_onsets(y, sr)
    cur = y
    times = midi = voiced = None
    for it in range(args.iterations):
        times, midi, voiced, _ = pyin_track_array(cur, sr)
        curve = np.where(voiced, midi, np.nan)
        cur = clean_with_prior(y, sr, times, curve)
        print(f"iteration {it+1}/{args.iterations} done", file=sys.stderr)
    notes = segment_notes(times, midi, voiced, bpm=args.bpm,
                           repeat_blocks=repeat_blocks, onsets=onsets)
    out = [[round(n["start"] + args.offset, 3), round(n["end"] + args.offset, 3), n["midi"]]
           for n in notes]
    ambiguous = [round(n["start"] + args.offset, 3) for n in notes if n["portamento_ambiguous"]]
    json.dump(out, open(args.out, "w"))
    print(f"{len(out)}音を書き出しました: {args.out}")
    if ambiguous:
        print(f"ポルタメント疑いで音高が不安定だった音の開始時刻: {ambiguous}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="正本がある区間で精度を測る")
    c.add_argument("vocals")
    c.add_argument("gold")
    c.add_argument("--offset", type=float, default=0.0)
    c.add_argument("--bpm", type=float, default=183.0)
    c.add_argument("--repeat-block", action="append", default=[])
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("transcribe", help="未知区間を自己参照浄化して書き起こす")
    t.add_argument("vocals")
    t.add_argument("--offset", type=float, default=0.0)
    t.add_argument("--bpm", type=float, default=183.0)
    t.add_argument("--iterations", type=int, default=2)
    t.add_argument("--repeat-block", action="append", default=[])
    t.add_argument("--out", default="candidates_auto.json")
    t.set_defaults(func=cmd_transcribe)

    args = ap.parse_args()
    args.func(args)
