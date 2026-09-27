"""パイプラインの回帰チェック(合成音声を使う。元音源が無くても実行できる)。

実際にこれで2つのバグを発見・修正した:
  1. repeat_block内で「無声の谷をまたがない」判定が、通常の連続フレームの継続まで
     邪魔してしまい、有声フラグが一度も落ちない反復句が丸ごと1音に潰れていた
  2. 隣接同音マージの判定を「マージ済みノートの元のstart」で見ていたため、ブロック
     境界をまたいで1回マージが起きた瞬間、以後の反復音が全部そこに吸収され続けていた

python3 pipeline/selftest.py で実行。異常があればAssertionErrorで落ちる。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from pitch_track import pyin_track_array, detect_onsets
from informed_separate import clean_with_prior, notes_to_pitch_curve
from notes import segment_notes
from evaluate import frame_accuracy, note_level_prf

SR = 22050


def synth_notes(notes, sr=SR):
    total_dur = max(e for _, e, _ in notes) + 0.2
    y = np.zeros(int(total_dur * sr))
    t_full = np.arange(len(y)) / sr
    for s, e, p in notes:
        freq = 440.0 * 2 ** ((p - 69) / 12.0)
        mask = (t_full >= s) & (t_full < e)
        n = mask.sum()
        if n == 0:
            continue
        tt = np.arange(n) / sr
        env = np.ones(n)
        ramp = min(200, n // 4)
        if ramp > 0:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        tone = 0.5 * np.sin(2 * np.pi * freq * tt) + 0.15 * np.sin(2 * np.pi * 2 * freq * tt)
        y[mask] += tone * env
    return y


# 88-106秒のようなケースを模した合成テストケース: 普通の音3つ + 反復句(8分音符間隔で
# 8音)+ 最後にもう1音。さらに最初の1.6秒に無関係な「ピアノ漏れ」の音を混ぜる
GOLD = [
    (0.0, 0.5, 65), (0.5, 1.0, 63), (1.0, 1.6, 68),
    (2.0, 2.14, 70), (2.164, 2.30, 70), (2.328, 2.46, 70), (2.492, 2.62, 70),
    (2.656, 2.80, 70), (2.82, 2.96, 70), (2.984, 3.12, 70), (3.148, 3.28, 70),
    (3.5, 4.0, 60),
]
REPEAT_BLOCKS = [(2.0, 3.3)]


def build_audio():
    y = synth_notes(GOLD)
    y_noisy = y + 0.02 * np.random.randn(len(y))
    t_full = np.arange(len(y)) / SR
    piano = np.zeros(len(y))
    mask_piano = t_full < 1.6
    piano[mask_piano] = 0.4 * np.sin(2 * np.pi * 300.0 * t_full[mask_piano])
    return y_noisy + piano


def run():
    y = build_audio()

    times, midi, voiced, _ = pyin_track_array(y, SR)
    onsets = detect_onsets(y, SR)

    naive = segment_notes(times, midi, voiced, bpm=183.0, repeat_blocks=REPEAT_BLOCKS, onsets=onsets)
    naive_notes = [(n["start"], n["end"], n["midi"]) for n in naive]
    print("naive:", frame_accuracy(naive_notes, GOLD), note_level_prf(naive_notes, GOLD))

    curve_times = np.arange(0, times[-1], 0.01)
    curve = notes_to_pitch_curve(GOLD, curve_times)
    y_clean = clean_with_prior(y, SR, curve_times, curve)
    times2, midi2, voiced2, _ = pyin_track_array(y_clean, SR)
    onsets2 = detect_onsets(y_clean, SR)
    cleaned = segment_notes(times2, midi2, voiced2, bpm=183.0, repeat_blocks=REPEAT_BLOCKS, onsets=onsets2)
    cleaned_notes = [(n["start"], n["end"], n["midi"]) for n in cleaned]
    print("cleaned:", frame_accuracy(cleaned_notes, GOLD), note_level_prf(cleaned_notes, GOLD))

    repeat_region = [n for n in cleaned_notes if 2.0 <= n[0] < 3.3]
    print(f"repeat block: {len(repeat_region)} notes found (gold has 8)")

    naive_acc = frame_accuracy(naive_notes, GOLD)
    cleaned_acc = frame_accuracy(cleaned_notes, GOLD)
    assert len(repeat_region) >= 6, (
        "反復句がまとめて1音に潰れている(88-106秒バグの再発)。"
        f"見つかったのは{len(repeat_region)}音"
    )
    assert cleaned_acc["within_1_semitone"] >= naive_acc["within_1_semitone"], (
        "score-informedな浄化がむしろ精度を下げている"
    )
    assert cleaned_acc["within_1_semitone"] >= 0.9, "浄化後もピアノ漏れの影響が大きすぎる"
    print("selftest OK")


if __name__ == "__main__":
    run()
