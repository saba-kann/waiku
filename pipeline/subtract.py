"""オフボーカル(インスト版)音源があるなら、Demucsより先にこちらを試す。

原理は単純な位相反転差分: ボーカル = 原曲 - k * オフボーカル。
Demucsのような機械学習の推定ではなく厳密な引き算なので、うまく時刻合わせと
音量合わせさえできれば、ML分離よりノイズが少ない(はずだが、マスタリングの違い
(リミッター等)で完全には消えない残留成分は出る)。

2026-09-27、実音源での検証: 原曲241.2秒・オフボーカル241.3秒(0.062秒のズレを
自動検出して補正)に対し、0〜132秒の正本と突き合わせたところ:
  浄化なし(この差分音声に直接pYIN)          : frame一致率(±1半音) 70.7%, ノートF1 0.31
  自己参照浄化(答えなし、bootstrap)          : frame一致率(±1半音) 71.2%, ノートF1 0.28
  正本で浄化(カンニングあり・上限の目安)      : frame一致率(±1半音) 81.3%, ノートF1 0.41
Demucsが使えないこの環境でも、ここまでは到達できる。ただし「楽譜情報で分離を改善した
原曲ボーカルでは96%」にはまだ届いていないので、Demucsの分離がさらに効く余地がある
(オフボーカル音源はこの曲固有に手に入ったもので、どの曲にも使える保証はない。
 一般化を考えるとDemucs系のパスも維持する)。
"""
import numpy as np
import librosa


def find_offset(orig, offv, sr, search_sec=20, window_sec=20):
    """origの先頭window_sec秒が、offvのどこに一番よく相関するかをサンプル単位で返す。
       返り値が正なら、offvはorigよりその分だけ遅れている(offv[offset:]で頭を揃える)。"""
    seg = orig[: int(window_sec * sr)]
    search = offv[: int((window_sec + search_sec) * sr)]
    corr = np.correlate(search - search.mean(), seg - seg.mean(), mode="valid")
    return int(np.argmax(corr))


def subtract_vocal(orig, offv, sr, offset=None):
    """orig, offv: 同じ曲のモノラル波形(原曲・オフボーカル)。
       戻り値: (残差=推定ボーカル, 使ったoffset, 使ったscale係数k)"""
    if offset is None:
        offset = find_offset(orig, offv, sr)
    aligned = offv[offset:] if offset >= 0 else np.concatenate(
        [np.zeros(-offset), offv])
    n = min(len(orig), len(aligned))
    orig_c, offv_c = orig[:n], aligned[:n]
    k = float(np.dot(orig_c, offv_c) / np.dot(offv_c, offv_c))
    residual = orig_c - k * offv_c
    return residual, offset, k


def subtract_vocal_files(orig_path, offvocal_path, sr=22050):
    orig, _ = librosa.load(orig_path, sr=sr, mono=True)
    offv, _ = librosa.load(offvocal_path, sr=sr, mono=True)
    return subtract_vocal(orig, offv, sr)


def denoise(y, sr, noise_start, noise_end, n_fft=2048, hop_length=512,
            over_subtract=1.5, floor=0.1):
    """原曲・オフボーカルがそれぞれ別々にMP3エンコードされていると、量子化ノイズが
    一致せず引き算で消えない(実測: 休符でもRMSが歌唱部の9割近く残る「砂嵐」状態)。
    休符区間(noise_start〜noise_end秒、本当に無音/無歌唱な区間を選ぶこと)からノイズの
    周波数プロファイルを推定し、スペクトル減算で下げる。

    実測(2026-09-27): 休符RMSが歌唱部の88%→68%まで改善(完全には消えない)。
    ノート単位F1はわずかに悪化(0.31→0.28程度)する一方、frame単位のピッチ一致率は
    改善した(70.7%→74.7%)。**人間が聴いて確認する用途(レビューツールの音声)には
    こちらを使う。ノーツの自動書き起こし自体は浄化前の音声で行った方がわずかに良い**
    (over_subtractを上げすぎるとmusical noiseが増えて逆効果になるので注意)"""
    noise_seg = y[int(noise_start * sr): int(noise_end * sr)]
    S_noise = np.abs(librosa.stft(noise_seg, n_fft=n_fft, hop_length=hop_length))
    noise_profile = np.median(S_noise, axis=1, keepdims=True)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    mag_clean = np.maximum(mag - over_subtract * noise_profile, floor * mag)
    S_clean = mag_clean * np.exp(1j * phase)
    return librosa.istft(S_clean, hop_length=hop_length, length=len(y))


if __name__ == "__main__":
    import sys
    import soundfile as sf
    residual, offset, k = subtract_vocal_files(sys.argv[1], sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "vocal_isolated.wav"
    sf.write(out, residual.astype(np.float32), 22050)
    print(f"offset={offset}samples k={k:.4f} -> {out}")
