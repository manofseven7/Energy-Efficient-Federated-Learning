"""Prepare auditable UCI HAR and CWRU arrays (PAMAP2 kept as optional legacy support).

Intel Lab preprocessing is implemented separately in ``preprocess_intel.py``
because it uses chronological next-observation regression targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "extracted"
OUT = ROOT / "data" / "processed"


def standardize(xtr, xte):
    mean = np.nanmean(xtr, axis=0)
    std = np.nanstd(xtr, axis=0)
    std[std < 1e-8] = 1.0
    return ((xtr - mean) / std).astype(np.float32), ((xte - mean) / std).astype(np.float32), mean, std


def save(name, xtr, y1tr, y2tr, xte, y1te, y2te, groups_tr, groups_te, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, xte, mean, std = standardize(xtr, xte)
    path = OUT / f"{name}.npz"
    np.savez_compressed(path, x_train=xtr, y1_train=y1tr.astype(np.int64),
                        y2_train=y2tr.astype(np.int64), x_test=xte,
                        y1_test=y1te.astype(np.int64), y2_test=y2te.astype(np.int64),
                        group_train=np.asarray(groups_tr), group_test=np.asarray(groups_te),
                        feature_mean=mean, feature_std=std,
                        metadata=json.dumps(meta, sort_keys=True))
    return {"dataset": name, "path": str(path), "train": len(xtr), "test": len(xte),
            "features": xtr.shape[1], "task1_classes": int(np.max(y1tr)+1),
            "task2_classes": int(np.max(y2tr)+1),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_uci_har():
    base = EXT / "uci_har" / "UCI HAR Dataset"
    def read(split):
        x = np.loadtxt(base / split / f"X_{split}.txt", dtype=np.float32)
        y = np.loadtxt(base / split / f"y_{split}.txt", dtype=np.int64) - 1
        s = np.loadtxt(base / split / f"subject_{split}.txt", dtype=np.int64)
        # Original activity ids 1--3 dynamic, 4--6 static.
        y2 = (y >= 3).astype(np.int64)
        return x, y, y2, s
    tr, te = read("train"), read("test")
    meta = {"source": "UCI HAR official split", "task2": "dynamic(activities 1-3) vs static(4-6)",
            "leakage_control": "official subject-disjoint split"}
    return save("uci_har", *tr[:3], *te[:3], tr[3], te[3], meta)


PAMAP_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PAMAP_MAP = {v:i for i,v in enumerate(PAMAP_ACTIVITIES)}
PAMAP_FEATURE_COLS = list(range(4, 16)) + list(range(21, 33)) + list(range(38, 50))


def read_pamap_subject(path, stride=10):
    # Streaming avoids holding the full 100-Hz text file in memory.
    xs, ya, hr = [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            vals = line.split()
            if len(vals) < 54:
                continue
            act = int(float(vals[1]))
            if act not in PAMAP_MAP:
                continue
            h = float(vals[2])
            if not np.isfinite(h):
                continue
            feat = np.array([float(vals[j]) for j in PAMAP_FEATURE_COLS], dtype=np.float32)
            xs.append(feat); ya.append(PAMAP_MAP[act]); hr.append(h)
    x = np.vstack(xs)
    # Column medians are fitted per subject only for missing sensor samples.
    med = np.nanmedian(x, axis=0)
    bad = np.where(~np.isfinite(x)); x[bad] = med[bad[1]]
    return x, np.asarray(ya), np.asarray(hr)


def prepare_pamap2():
    base = EXT / "pamap2" / "PAMAP2_Dataset" / "Protocol"
    train_ids = list(range(101, 107)); test_ids = list(range(107, 110))
    cache = {}
    for sid in train_ids + test_ids:
        cache[sid] = read_pamap_subject(base / f"subject{sid}.dat")
    train_hr = np.concatenate([cache[s][2] for s in train_ids])
    q1, q2 = np.quantile(train_hr, [1/3, 2/3])
    def combine(ids):
        x=np.vstack([cache[s][0] for s in ids]); y1=np.concatenate([cache[s][1] for s in ids])
        h=np.concatenate([cache[s][2] for s in ids]); y2=np.digitize(h, [q1,q2])
        g=np.concatenate([np.full(len(cache[s][1]),s) for s in ids])
        return x,y1,y2,g
    tr,te=combine(train_ids),combine(test_ids)
    meta={"source":"PAMAP2 Protocol","sampling":"100 Hz downsampled to 10 Hz",
          "train_subjects":train_ids,"test_subjects":test_ids,"heart_rate_thresholds": [float(q1),float(q2)],
          "leakage_control":"heart-rate thresholds and standardization fitted on training subjects only"}
    return save("pamap2", *tr[:3], *te[:3], tr[3], te[3], meta)


CWRU_FILES = {
    0: {"normal": [97,98,99,100]},
    1: {"inner":[105,106,107,108], "ball":[118,119,120,121], "outer":[130,131,132,133]},
    2: {"inner":[169,170,171,172], "ball":[185,186,187,188], "outer":[197,198,199,200]},
    3: {"inner":[209,210,211,212], "ball":[222,223,224,225], "outer":[234,235,236,237]},
}
CWRU_TYPE = {"normal":0,"inner":1,"ball":2,"outer":3}


def cwru_signal(file_id):
    m=loadmat(RAW/"cwru"/f"{file_id}.mat")
    candidates=[np.ravel(v) for k,v in m.items() if "DE_time" in k]
    if not candidates:
        raise KeyError(f"No DE_time channel in {file_id}.mat: {list(m)}")
    return candidates[0].astype(np.float64)


def window_features(x, n=2048):
    count=len(x)//n; x=x[:count*n].reshape(count,n)
    mu=x.mean(1); sd=x.std(1)+1e-12; rms=np.sqrt(np.mean(x*x,1)); ptp=np.ptp(x,axis=1)
    sk=skew(x,axis=1,bias=False); ku=kurtosis(x,axis=1,bias=False)
    crest=np.max(np.abs(x),axis=1)/rms; zcr=np.mean(np.diff(np.signbit(x),axis=1),axis=1)
    spec=np.abs(np.fft.rfft(x,axis=1))**2 + 1e-12
    bands=np.array_split(np.arange(spec.shape[1]),16)
    loge=np.stack([np.log1p(spec[:,b].mean(1)) for b in bands],axis=1)
    p=spec/spec.sum(1,keepdims=True); centroid=(p*np.linspace(0,1,p.shape[1])).sum(1)
    entropy=-(p*np.log(p)).sum(1)/np.log(p.shape[1])
    return np.column_stack([mu,sd,rms,ptp,sk,ku,crest,zcr,centroid,entropy,loge])


def prepare_cwru():
    tr, te = [], []
    for sev, groups in CWRU_FILES.items():
        for typ, ids in groups.items():
            for load_idx,fid in enumerate(ids):
                feat=window_features(cwru_signal(fid))
                item=(feat,np.full(len(feat),CWRU_TYPE[typ]),np.full(len(feat),sev),np.full(len(feat),fid))
                # Loads 0--2 train; load 3 test. Entire recordings stay in one split.
                (te if load_idx==3 else tr).append(item)
    def combine(items): return tuple(np.concatenate([z[i] for z in items]) for i in range(4))
    tr,te=combine(tr),combine(te)
    meta={"source":"CWRU official 12-kHz drive-end files","window":2048,"overlap":0,
          "train_loads_hp":[0,1,2],"test_load_hp":3,"leakage_control":"recording/load-disjoint test split"}
    return save("cwru", *tr[:3], *te[:3], tr[3], te[3], meta)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--datasets",nargs="+",default=["uci_har","cwru"])
    args=ap.parse_args(); reports=[]
    for name in args.datasets:
        reports.append(globals()[f"prepare_{name}"]())
        print(json.dumps(reports[-1]))
    (OUT/"manifest.json").write_text(json.dumps(reports,indent=2))

if __name__ == "__main__": main()
