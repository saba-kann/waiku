#!/usr/bin/env python3
"""Synthesia動画の落下ノーツから右手(緑)の音符を抽出する。

使い方: python3 extract.py 動画.mp4  -> candidates.json を出力

重要な設定(すべて検証済み。変えると除外リストと一致しなくなる):
  - 黒鍵は画像から探さず、52本の白鍵の境界から幾何的に置く。
    画像検出だと黒鍵を1つ見落とし、B♭2以上が全部1半音低くなった。
  - 落下速度 SPEED=300 px/秒(固定値)。自動計測は285と出て時刻がずれた。
  - OFFSET=1.97 秒を引いて原曲MP3の時刻に合わせる。
  - この設定で右手2186音、既存の除外リストと254件中252件(99%)一致。
"""
import json
import sys

import cv2
import numpy as np

TOP, KEY_LINE, SPEED, FPS, OFFSET = 280, 850, 300.0, 30.0, 1.97


def white_keys(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    bright = np.array([np.mean(gray[y]) > 120 for y in range(h)])
    y = h - 1
    while y > 0 and bright[y]:
        y -= 1
    kb = gray[y + 1:]
    row = kb[int(kb.shape[0] * 0.9)]
    dark = row < 170
    bounds, inr, s = [], False, 0
    for x, d in enumerate(dark):
        if d and not inr:
            inr, s = True, x
        elif not d and inr:
            inr = False
            bounds.append((s + x) // 2)
    if inr:
        bounds.append((s + len(dark)) // 2)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def keymap(whites):
    wm = [m for m in range(21, 109) if m % 12 not in (1, 3, 6, 8, 10)]
    assert len(whites) == 52, f"白鍵が52本でない: {len(whites)}"
    km = {m: whites[i] for i, m in enumerate(wm)}
    for i in range(len(wm) - 1):
        if wm[i + 1] - wm[i] == 2:          # 間に黒鍵がある
            b = whites[i][1]
            w = whites[i][1] - whites[i][0]
            km[wm[i] + 1] = (int(b - w * 0.3), int(b + w * 0.3))
    assert len(km) == 88
    return km


def extract(video):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    _, frame = cap.read()
    km = keymap(white_keys(frame))
    centers = sorted(((a + b) / 2, m) for m, (a, b) in km.items())
    cx = np.array([c for c, _ in centers])
    cm = np.array([m for _, m in centers])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw = []
    for fi in range(0, total, int(1.5 * FPS)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(f[TOP:KEY_LINE], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green = (s > 90) & (v > 90) & (h >= 40) & (h <= 85)   # 右手
        n, _, st, ce = cv2.connectedComponentsWithStats(
            green.astype(np.uint8) * 255, connectivity=8)
        for i in range(1, n):
            x, y, w, hh, ar = st[i]
            if hh < 3 or w < 5 or ar < 30:
                continue
            midi = int(cm[np.argmin(np.abs(cx - ce[i][0]))])
            onset = fi / FPS + (KEY_LINE - (TOP + y + hh)) / SPEED
            if onset >= 0:
                raw.append((round(onset, 3), round(hh / SPEED, 3), midi))
    raw.sort(key=lambda r: (r[2], r[0]))
    out = []
    for o, d, m in raw:
        if out and out[-1][2] == m and abs(out[-1][0] - o) < 0.06:
            if d > out[-1][1]:
                out[-1] = [o, d, m]
            continue
        out.append([o, d, m])
    notes = sorted((round(o - OFFSET, 2), round(o + d - OFFSET, 2), m)
                   for o, d, m in out if o - OFFSET >= 0)
    return notes


if __name__ == "__main__":
    notes = extract(sys.argv[1])
    json.dump([list(x) for x in notes], open("candidates.json", "w"))
    print(f"右手 {len(notes)}音 -> candidates.json")
