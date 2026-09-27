"""連続ピッチ曲線(pYINの出力)を離散ノーツ([start, end, midi])に変換する。

build.pyの「80秒以降のトリルは同音が0.2秒以内で続けば1つの伸ばした音にまとめる」規則が
88〜106秒の歌詞反復句(個々の音として残すべきだった)を誤って壊した教訓を反映し、
「区間ごとに一律マージ」ではなく「repeat_blockかどうかで挙動を変える」設計にしている。
"""
import numpy as np


def segment_notes(times, midi, voiced,
                   bpm=183.0, semitone_jump=0.6, gap_bridge=0.03,
                   min_note_dur=0.03, adjacent_merge_gap=0.015,
                   repeat_blocks=(), snap_tolerance=0.04, onsets=()):
    """times, midi, voiced: pitch_track.pyin_track() の戻り値。
       repeat_blocks: [(start,end), ...] 歌詞反復句など、同音が近接して並ぶのが正しい区間。
         この区間内では隣接する同音符の再マージを行わない(88-106秒バグの再発防止)。
       onsets: pitch_track.detect_onsets() の出力。repeat_blocks内では、同じ高さ・
         無声区間なしの再アタックがピッチ/有声フラグだけでは検出できないことがあるため、
         オンセット検出の結果を強制的な音符境界として使う
       戻り値: [{"start":s, "end":e, "midi":m, "portamento_ambiguous":bool}, ...]
    """
    times = np.asarray(times)
    midi = np.asarray(midi, dtype=float)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(midi)
    hop = np.median(np.diff(times)) if len(times) > 1 else 0.01

    def in_repeat_block(t):
        return any(s <= t < e for s, e in repeat_blocks)

    # 1) 有声フレームを、短い無声の谷(gap_bridge以内)をまたいで連続runにまとめる。
    #    repeat_block内では絶対にまたがない(88-106秒バグの再発防止: 同じ高さの音が
    #    近接して繰り返す区間では、無声の谷=本物の再アタックである可能性が高いため)
    runs = []
    run_start = None
    last_voiced_i = None
    for i, v in enumerate(voiced):
        if v:
            if run_start is None:
                run_start = i
            elif last_voiced_i is not None and i - last_voiced_i > 1:
                # 直前のフレームがそのまま有声(i == last_voiced_i+1)なら、そもそも
                # またぐべきギャップがないので何もしない。無声フレームを挟んだ場合だけ判定する
                gap = times[i] - times[last_voiced_i]
                bridgeable = gap <= gap_bridge and not in_repeat_block(times[last_voiced_i])
                if not bridgeable:
                    runs.append((run_start, last_voiced_i))
                    run_start = i
            last_voiced_i = i
        # v=False: 何もしない(bridgeableならrunを継続扱い)
    if run_start is not None:
        runs.append((run_start, last_voiced_i))

    onsets = np.asarray(sorted(onsets), dtype=float)

    def onset_between(t_prev, t_cur):
        if len(onsets) == 0:
            return False
        return bool(np.any((onsets > t_prev) & (onsets <= t_cur)))

    raw_notes = []
    for a, b in runs:
        seg_start = a
        for i in range(a + 1, b + 1):
            if not voiced[i]:
                continue
            pitch_jump = abs(midi[i] - midi[i - 1]) > semitone_jump
            forced_reattack = (in_repeat_block(times[i])
                                and onset_between(times[i - 1], times[i]))
            if pitch_jump or forced_reattack:
                raw_notes.append(_finish_segment(times, midi, voiced, seg_start, i - 1, hop))
                seg_start = i
        raw_notes.append(_finish_segment(times, midi, voiced, seg_start, b, hop))

    raw_notes = [n for n in raw_notes if n is not None and (n["end"] - n["start"]) >= min_note_dur]

    # 2) 隣接同音マージ(repeat_block内は行わない。in_repeat_blockは上で定義済み)
    #    判定は「これからマージしようとしている新しい音(n)」の位置で行う。
    #    マージ済みノートの元のstartで判定すると、ブロック境界をまたいで1回マージが
    #    起きた瞬間に以後ずっと判定が狂う(境界直前の断片と最初の反復音が結合すると、
    #    そのstartがブロック外になり、以後の反復音が全部その1つにマージされ続けてしまう)
    merged = []
    for n in raw_notes:
        if (merged and merged[-1]["midi"] == n["midi"]
                and n["start"] - merged[-1]["end"] <= adjacent_merge_gap
                and not in_repeat_block(n["start"])):
            merged[-1]["end"] = n["end"]
        else:
            merged.append(n)

    # 3) BPMグリッドへの軽いスナップ(近ければ、の任意補正)
    beat = 60.0 / bpm
    grid = beat / 2.0  # 8分音符
    for n in merged:
        n["start"] = _snap(n["start"], grid, snap_tolerance)
        n["end"] = _snap(n["end"], grid, snap_tolerance)

    return merged


def _finish_segment(times, midi, voiced, i0, i1, hop):
    idx = [i for i in range(i0, i1 + 1) if voiced[i]]
    if not idx:
        return None
    pitches = midi[idx]
    med = float(np.median(pitches))
    rounded = int(round(med))
    # ポルタメント疑い: 区間内の音高が中央値から0.4半音以上ばらついている
    ambiguous = bool(np.max(np.abs(pitches - med)) > 0.4) if len(pitches) > 1 else False
    return {
        "start": float(times[idx[0]] - hop / 2),
        "end": float(times[idx[-1]] + hop / 2),
        "midi": rounded,
        "portamento_ambiguous": ambiguous,
    }


def _snap(t, grid, tol):
    nearest = round(t / grid) * grid
    return nearest if abs(nearest - t) <= tol else t
