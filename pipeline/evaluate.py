"""自動採譜パイプラインの結果を、信頼できる基準と突き合わせて評価する。

CLAUDE.mdの教訓(OMRデータの時間軸ズレ、ピアノ編曲を「歌」として使ったこと)を繰り返さないため、
基準として使うのは以下のうち検証済みのものだけにすること:
  - data/vocal_melody_0_132s.json (ユーザー確認済みの正本。最優先)
  - カバー歌手の実歌唱(data/cover_pitch_0_84s.json 等、人間の声)
ピアノ編曲の右手(data/video_right_hand_notes.json)を「歌の正解」として使わないこと。
"""
import numpy as np


def frame_accuracy(pred_notes, gold_notes, t0=None, t1=None, hop=0.02):
    """両方をhop秒間隔でサンプルし、有声(gold側に音がある)フレームでの一致率を返す。"""
    gold_notes = sorted(gold_notes)
    pred_notes = sorted(pred_notes)
    if t0 is None:
        t0 = gold_notes[0][0] if gold_notes else 0.0
    if t1 is None:
        t1 = gold_notes[-1][1] if gold_notes else 0.0
    ts = np.arange(t0, t1, hop)

    def pitch_at(notes, t):
        for s, e, p in notes:
            if s <= t < e:
                return p
        return None

    exact = within1 = total = 0
    for t in ts:
        g = pitch_at(gold_notes, t)
        if g is None:
            continue
        p = pitch_at(pred_notes, t)
        total += 1
        if p is not None:
            if p == g:
                exact += 1
            if abs(p - g) <= 1:
                within1 += 1
    if total == 0:
        return {"n_frames": 0, "exact": None, "within_1_semitone": None}
    return {
        "n_frames": total,
        "exact": exact / total,
        "within_1_semitone": within1 / total,
    }


def note_level_prf(pred_notes, gold_notes, onset_tol=0.08):
    """gold各音について、同じ音高・近い開始時刻の予測音があるかを見て precision/recall/F1 を出す。"""
    gold_notes = list(gold_notes)
    pred_notes = list(pred_notes)
    matched_pred = set()
    tp = 0
    for gs, ge, gp in gold_notes:
        hit = None
        for i, (ps, pe, pp) in enumerate(pred_notes):
            if i in matched_pred:
                continue
            if pp == gp and abs(ps - gs) <= onset_tol:
                hit = i
                break
        if hit is not None:
            matched_pred.add(hit)
            tp += 1
    precision = tp / len(pred_notes) if pred_notes else 0.0
    recall = tp / len(gold_notes) if gold_notes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "n_pred": len(pred_notes), "n_gold": len(gold_notes),
            "precision": precision, "recall": recall, "f1": f1}
