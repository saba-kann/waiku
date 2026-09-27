"""音声からフレームごとの基本周波数(f0)を追跡する。pYIN(librosa)を使う。

このモジュールはネットワーク接続を必要としない(モデルのダウンロード不要)。
分離済みボーカルのwavに対して使う想定。
"""
import numpy as np
import librosa


def pyin_track(wav_path, fmin="C2", fmax="C7", frame_length=2048, hop_length=256, sr=None):
    """戻り値: times(秒), midi(半音, 無声区間はnan), voiced_flag(bool), voiced_prob(0-1)"""
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    fmin_hz = librosa.note_to_hz(fmin) if isinstance(fmin, str) else fmin
    fmax_hz = librosa.note_to_hz(fmax) if isinstance(fmax, str) else fmax
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin_hz, fmax=fmax_hz,
        frame_length=frame_length, hop_length=hop_length, sr=sr,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(f0 / 440.0)
    return times, midi, voiced_flag.astype(bool), voiced_prob


def detect_onsets(y, sr, hop_length=256):
    """アタック(音の立ち上がり)の時刻を検出する。同じ高さが近接して繰り返す区間
    (歌詞の反復句など)では、無声区間なしに再アタックすることがあり、
    ピッチ・有声フラグだけでは音符の境目が分からない。そこで使う補助情報。"""
    return librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units="time",
                                       backtrack=False)


def pyin_track_array(y, sr, fmin="C2", fmax="C7", frame_length=2048, hop_length=256):
    """wavファイルではなく既にメモリ上にある波形(numpy配列)に対して使う版。
       informed_separate.py の反復処理(浄化→再追跡)で使う。"""
    fmin_hz = librosa.note_to_hz(fmin) if isinstance(fmin, str) else fmin
    fmax_hz = librosa.note_to_hz(fmax) if isinstance(fmax, str) else fmax
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin_hz, fmax=fmax_hz,
        frame_length=frame_length, hop_length=hop_length, sr=sr,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(f0 / 440.0)
    return times, midi, voiced_flag.astype(bool), voiced_prob
