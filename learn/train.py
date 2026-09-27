"""label_store.jsonl から merge/keep_separate の分類器を学習する。
   件数が少ない前提で単純なロジスティック回帰を使う。uncertain は除外する。"""
import json
import numpy as np

FEATS = ["duration", "n_notes", "pitch_range", "returns_to_start",
         "rate_hz", "is_repeat_block", "voice_smooth"]

def load(path="label_store.jsonl"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r["label"] in ("merge", "keep_separate")]

def to_xy(rows):
    X = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "keep_separate" else 0 for r in rows])
    return X, y

def train():
    rows = load()
    if len(rows) < 6:
        print(f"教師データが{len(rows)}件しかありません。"
              f"最低6件(両方のラベルを含む)集まってから学習してください。")
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = to_xy(rows)
    if len(set(y)) < 2:
        print("ラベルが一種類しかありません。両方の例が必要です。")
        return None
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)
    print(f"{len(rows)}件で学習しました。")
    return clf, sc

def predict_confidence(clf, sc, feat):
    x = np.array([[feat[f] for f in FEATS]])
    p = clf.predict_proba(sc.transform(x))[0, 1]
    label = "keep_separate" if p > 0.5 else "merge"
    confidence = abs(p - 0.5) * 2   # 0=五分五分, 1=確信
    return label, confidence

if __name__ == "__main__":
    train()
