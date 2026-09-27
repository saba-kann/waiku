"""揺れ区間から特徴量を作る。mel_*.json の一部と、対応する声のピッチ列(t,f0,voiced)を渡す。"""
import numpy as np

def extract(notes, t, m, v, is_repeat_block=False):
    """notes: [(start,end,pitch), ...] 連続した短い音の並び
       t,m,v: その区間を覆う時刻・声の音高(半音)・有声フラグの配列"""
    s, e = notes[0][0], notes[-1][1]
    duration = e - s
    n_notes = len(notes)
    pitches = [p for _, _, p in notes]
    pitch_range = max(pitches) - min(pitches)
    returns_to_start = int(notes[0][2] == notes[-1][2])
    rate_hz = n_notes / duration if duration > 0 else 0.0

    k = (t >= s) & (t < e) & v
    if k.sum() >= 3:
        mm = m[k]
        # 実際の声が「段差」か「なめらか」かを、二階差分の小ささで見る
        d1 = np.diff(mm)
        d2 = np.diff(d1)
        voice_smooth = float(1.0 / (1.0 + np.std(d2) * 5))
    else:
        voice_smooth = 0.5  # 声の証拠が薄い場合は中立

    return dict(duration=round(duration, 3), n_notes=n_notes,
                pitch_range=pitch_range, returns_to_start=returns_to_start,
                rate_hz=round(rate_hz, 2), is_repeat_block=int(is_repeat_block),
                voice_smooth=round(voice_smooth, 3))
