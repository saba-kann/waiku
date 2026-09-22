#!/usr/bin/env python3
"""候補音(candidates.json)にユーザーの除外指定を適用し、旋律MIDIを作る。

使い方: python3 build.py [確認済み範囲の終わり秒=112]
  入力: data/video_right_hand_notes.json(候補音)、exclusions.json、
        data/cover_pitch_0_84s.json(カバー歌手の音高。60秒以降のオクターブ補正に使う)
  出力: melody.mid / melody.json

後処理の規則(前セッションで確定したもの):
  0-60秒   : 除外を適用して単旋律化するだけ
  60-96秒  : カバー歌手の音高に対し ±7半音に収まるようオクターブ単位で寄せ、
             それでも4半音以上離れる音は除去
  84-88秒  : カバーが無声なので、歌手の80-84秒の中央値(F5)を基準に同様に寄せる
  80秒以降 : トリル区間。ユーザーは往復の下の音を外し上の音を残している。
             残った同音が0.20秒以内で続くものは1つの伸ばした音にまとめる
  100.08秒 : 30msのB4は抽出誤差として除去
"""
import json
import sys

import numpy as np


def load_exclusions(path="exclusions.json"):
    ex = json.load(open(path))
    return [(d["t"], d["midi"]) for v in ex.values() for d in v]


def apply_exclusions(cands, ex):
    drop = set()
    for t, m in ex:
        for i, (s, e, p) in enumerate(cands):
            if i not in drop and abs(s - t) < 0.03 and p == m:
                drop.add(i)
                break
    return [c for i, c in enumerate(cands) if i not in drop]


def monophonic(mel):
    out = []
    for s, e, p in sorted(mel):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], s, out[-1][2])
        out.append((s, e, p))
    return [(s, e, p) for s, e, p in out if e - s > 0.02]


def fold(p, ref):
    while p - ref > 7:
        p -= 12
    while ref - p > 7:
        p += 12
    return p


def write_midi(mel, path, bpm=183.0):
    from music21 import stream, note, tempo, meter, key, instrument
    beat = 60.0 / bpm
    part = stream.Part()
    for s, e, p in mel:
        n = note.Note(p)
        n.duration.quarterLength = max(0.0625, (e - s) / beat)
        part.insert(s / beat, n)
    part.quantize((8,), processOffsets=True, processDurations=True, inPlace=True)
    part.insert(0, instrument.Vocalist())
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, key.Key("D-"))
    part.insert(0, tempo.MetronomeMark(number=bpm))
    sc = stream.Score()
    sc.insert(0, part)
    sc.write("midi", fp=path)


if __name__ == "__main__":
    LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 112.0   # ユーザー確認済みの範囲
    cands = [tuple(x) for x in json.load(open("data/video_right_hand_notes.json"))
             if x[0] < LIMIT]
    kept = apply_exclusions(cands, load_exclusions())
    print(f"候補 {len(cands)} -> 除外後 {len(kept)}")

    sung = None
    cp = json.load(open("data/cover_pitch_0_84s.json"))
    ct = np.array(cp["t"], dtype=float)
    cf = np.array([np.nan if x is None else x for x in cp["f0_hz"]], dtype=float)
    cv = np.array(cp["voiced"], dtype=bool) & ~np.isnan(cf)
    if True:

        def sung(s, e, w=0.15):
            m = (ct >= s - w) & (ct <= e + w) & cv
            if m.sum() < 2:
                return None
            return float(np.median(69 + 12 * np.log2(cf[m] / 440.0)))

    out = []
    for s, e, p in kept:
        if s >= 60 and sung is not None:
            v = sung(s, e)
            if v is None and 84 <= s < 88:
                v = 76.9                     # 歌手80-84秒の中央値 F5
            if v is not None:
                p = fold(p, v)
                if abs(p - v) > 4 and not (84 <= s < 88):
                    continue
        out.append((s, e, p))
    out = [x for x in out if not (abs(x[0] - 100.08) < 0.02 and x[2] == 71)]

    merged = []
    for s, e, p in sorted(out):
        if merged and merged[-1][2] == p and (
                (s >= 80 and s - merged[-1][1] <= 0.20)
                or abs(s - merged[-1][0]) < 0.03):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), p)
        else:
            merged.append((s, e, p))
    mel = monophonic(merged)
    print(f"旋律 {len(mel)}音")
    write_midi(mel, "melody.mid")
    json.dump([[round(s, 3), round(e, 3), int(p)] for s, e, p in mel],
              open("melody.json", "w"))
