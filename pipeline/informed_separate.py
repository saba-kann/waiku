"""「楽譜情報で分離を改善する」の具体的な実装(score-informed separation)。

前セッションで「楽譜情報で分離を改善した原曲ボーカルではpYINの旋律一致率96%」という
結果だけが伝わり、手順は失われていた。このモジュールは、その手順を正確に再現する
のではなく、**手元で検証可能な情報だけを使って**同種の効果を狙う。

やっていること:
Demucsの"vocals"ステムにはピアノが漏れている(CLAUDE.md記載: イントロで曲全体平均の2倍)。
その音声のSTFTに対し、「その時刻に鳴っているはずの声の高さ」の倍音付近だけを残す
時間変化ハーモニックマスクをかけ、ピアノなど無関係な周波数成分を減衰させる。

「その時刻に鳴っているはずの声の高さ」の与え方(信頼できる順):
1. gold: ユーザー確認済みの正本(data/vocal_melody_0_132s.json など)。0〜132秒の検証・
   キャリブレーションに使う。これが一番正確
2. self: 浄化前の音声に一度pYINをかけた自己推定(bootstrap)。正本がない区間
   (165〜241秒)で使う。自己参照なので鶏と卵だが、ハーモニックマスクは「今推定した
   基本周波数の倍音以外を削る」だけなので、推定が大まかに合っていれば
   ピアノの周波数(声の倍音上に乗っていない音)は削れる。1〜2回反復すると安定する
"""
import numpy as np
import librosa


def notes_to_pitch_curve(notes, times):
    """notes: [(start,end,midi), ...] を times(秒配列)上の連続MIDI値に変換。
       音がない時刻はnan(=マスクをかけない/素通しの目印)。"""
    midi = np.full_like(times, np.nan, dtype=float)
    notes = sorted(notes)
    for s, e, p in notes:
        k = (times >= s) & (times < e)
        midi[k] = p
    return midi


def harmonic_mask(mag, freqs, midi_curve, n_harmonics=8, bandwidth_cents=45, floor=0.12):
    """mag: STFTの振幅 shape (n_freq, n_frames)。freqs: 各binのHz。
       midi_curve: 各フレームの期待MIDI値(nanなら素通し=マスク1.0)。
       戻り値: 同shapeのマスク(0〜1)。"""
    n_freq, n_frames = mag.shape
    mask = np.ones_like(mag)
    bw = 2 ** (bandwidth_cents / 1200.0) - 1  # 相対帯域幅
    for i in range(n_frames):
        m = midi_curve[i]
        if not np.isfinite(m):
            continue
        f0 = 440.0 * 2 ** ((m - 69) / 12.0)
        col = np.full(n_freq, floor)
        for h in range(1, n_harmonics + 1):
            fh = f0 * h
            if fh > freqs[-1]:
                break
            sigma = fh * bw
            col = np.maximum(col, np.exp(-0.5 * ((freqs - fh) / sigma) ** 2))
        mask[:, i] = col
    return mask


def clean_with_prior(y, sr, midi_curve_times, midi_curve, n_fft=2048, hop_length=256,
                      n_harmonics=8, bandwidth_cents=45, floor=0.12):
    """音声yを、期待ピッチ曲線(midi_curve_times, midi_curve)でハーモニックマスクして返す。
       midi_curveの時間解像度がSTFTのフレームと違ってもよい(補間する)。"""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)
    curve_on_frames = np.interp(
        frame_times, midi_curve_times, midi_curve,
        left=np.nan, right=np.nan,
    )
    valid = ~np.isnan(midi_curve)
    if valid.sum() >= 2:
        # nanを挟む区間はnp.interpが直線補間してしまうので、near側にnanが無い場合だけ有効値を使う
        nearest_gap = np.interp(frame_times, midi_curve_times[valid], midi_curve_times[valid])
        gap = np.abs(frame_times - nearest_gap)
        curve_on_frames = np.where(gap < 0.25, curve_on_frames, np.nan)
    mask = harmonic_mask(mag, freqs, curve_on_frames, n_harmonics, bandwidth_cents, floor)
    S_clean = (mag * mask) * np.exp(1j * phase)
    return librosa.istft(S_clean, hop_length=hop_length, length=len(y))


def clean_self_informed(y, sr, fmin="C2", fmax="C7", iterations=2, **kwargs):
    """正本がない区間用: 自分自身のpYIN推定を反復的にプライアとして使う。"""
    from pitch_track import pyin_track_array
    cur = y
    times = midi = None
    for _ in range(max(1, iterations)):
        times, midi, voiced, _ = pyin_track_array(cur, sr, fmin=fmin, fmax=fmax)
        midi_masked = np.where(voiced, midi, np.nan)
        cur = clean_with_prior(y, sr, times, midi_masked, **kwargs)
    return cur, times, midi
