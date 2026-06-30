"""
CSVイベントファイルをNPYキャッシュに変換するスクリプト（一度だけ実行）
実行後は A_MIL_bag.py が自動でキャッシュを利用する。
"""
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

MAX_LEN     = 1000
NUM_WORKERS = 8

folders = {
    "101": (r"C:\bag\101\EWMAuto_events", 0),
    "102": (r"C:\bag\102\EWMAuto_events", 0),
    "103": (r"C:\bag\103\EWMAuto_events", 0),
    "401": (r"C:\bag\401\EWMAuto_events", 1),
    "402": (r"C:\bag\402\EWMAuto_events", 1),
    "403": (r"C:\bag\403\EWMAuto_events", 1),
    "301": (r"C:\bag\301\EWMAuto_events", 0),
    "302": (r"C:\bag\302\EWMAuto_events", 0),
    "303": (r"C:\bag\303\EWMAuto_events", 0),
    "601": (r"C:\bag\601\EWMAuto_events", 1),
    "602": (r"C:\bag\602\EWMAuto_events", 1),
    "603": (r"C:\bag\603\EWMAuto_events", 1),

}


def _load_one(filepath):
    try:
        df   = pd.read_csv(filepath)
        wave = (df['波形'].dropna().values if '波形' in df.columns
                else df.iloc[:, 0].dropna().values)
        if len(wave) == 0 or len(wave) > MAX_LEN:
            return None
        return (wave - wave[0]).astype(np.float32)
    except Exception:
        return None


def convert_folder(name, folder, label):
    cache_prefix = folder + "_cache"
    if os.path.exists(cache_prefix + "_data.npy"):
        print(f"[{name}] キャッシュ済み、スキップ: {cache_prefix}_data.npy")
        return

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    total = len(files)
    print(f"[{name}] {total:,} ファイルを {NUM_WORKERS} スレッドで並列ロード中...")

    data      = []
    processed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for wave in executor.map(_load_one, files):
            processed += 1
            if wave is not None:
                data.append(wave)
            if processed % 100000 == 0:
                print(f"  [{name}] {processed:,}/{total:,} 処理済み ({len(data):,} 件ロード)")

    np.save(cache_prefix + "_data.npy",
            np.array(data, dtype=object), allow_pickle=True)
    np.save(cache_prefix + "_labels.npy",
            np.array([label] * len(data), dtype=np.int64))
    print(f"[{name}] 完了: {len(data):,} 件 -> {cache_prefix}_*.npy\n")


if __name__ == "__main__":
    for name, (folder, label) in folders.items():
        convert_folder(name, folder, label)
    print("全フォルダの変換が完了しました。")
