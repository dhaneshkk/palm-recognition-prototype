"""Raw palm images → one uint8 ROI archive with identity labels.

    python training/prepare.py dataset/archive/session1 dataset/archive/session2 -o data/rois.npz

Uses preprocess/preprocessor.py, the code the Rust inference port reproduces bit-exactly, so training
sees exactly what inference sees. ROIs are stored as uint8: the normalised ROI is resized_uint8 / 255,
so this is lossless. Also accepts an existing folder of preprocessed .npy files (batch_preprocess.py).
"""
import argparse, json, os, subprocess

import numpy as np

from common import SIDE, identity, import_preprocessor, log, repo_root

EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="image folders (folder name = session) or one folder of <session>_<n>.npy ROIs")
    ap.add_argument("-o", "--out", default="data/rois.npz")
    ap.add_argument("--block", type=int, default=10, help="consecutive images per palm per session")
    ap.add_argument("--roi", choices=["centroid", "valley"], default="centroid",
                    help="centroid: the prototype's crop (preprocessor.py). valley: aligned to the finger valleys, "
                         "cut by the Rust binary so training and inference use the same code")
    ap.add_argument("--rust-bin", default=os.path.join(repo_root(), "rust", "target", "release", "palmprint"))
    ap.add_argument("--from-rust-rois", default=None, help="prefix of an existing `palmprint rois` export (skips running the binary)")
    a = ap.parse_args()
    if a.roi == "valley":
        return prepare_with_rust(a)
    pre = import_preprocessor()
    rois, names, failed = [], [], 0
    for d in a.dirs:
        folder = os.path.basename(os.path.normpath(d))
        files = sorted(os.listdir(d))
        for i, f in enumerate(files):
            stem, ext = os.path.splitext(f)
            p = os.path.join(d, f)
            if ext.lower() == ".npy":
                r, name = np.load(p), stem
            elif ext.lower() in EXT:
                r, name = pre(p), f"{folder}_{stem}"
            else:
                continue
            if r is None or r.shape != (SIDE, SIDE):
                failed += 1
                continue
            rois.append(np.round(r * 255.0).astype(np.uint8))
            names.append(name)
            if len(names) % 500 == 0:
                log("prepare", done=len(names), failed=failed, folder=folder, of=len(files))
    sess, ids = zip(*(identity(n, a.block) for n in names))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, rois=np.stack(rois), names=np.array(names), ids=np.array(ids, np.int64), sessions=np.array(sess), roi="centroid")
    log("prepared", images=len(names), failed=failed, palms=len(set(ids)), sessions=sorted(set(sess)), out=a.out, roi="centroid")


def prepare_with_rust(a):
    """Finger-valley crops from `palmprint rois` (rust/src/align.rs): one implementation for both sides."""
    if a.from_rust_rois:
        tmp, stats = a.from_rust_rois, json.load(open(a.from_rust_rois + ".stats.json"))
    else:
        if not os.path.exists(a.rust_bin):
            raise SystemExit(f"{a.rust_bin} not found: build it first (cd rust && cargo build --release) or pass --rust-bin")
        tmp = os.path.splitext(os.path.abspath(a.out))[0] + "_rust"
        cmd = [a.rust_bin, "rois", "--mode", "valley", "--out", tmp] + [os.path.abspath(d) for d in a.dirs]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"palmprint rois failed:\n{r.stderr[-2000:]}")
        stats = json.loads(r.stdout.strip().splitlines()[-1])
    rois = np.load(tmp + ".u8.npy").reshape(-1, SIDE, SIDE)
    names = [str(n) for n in np.load(tmp + ".names.npy")]
    sess, ids = zip(*(identity(n, a.block) for n in names))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, rois=rois, names=np.array(names), ids=np.array(ids, np.int64), sessions=np.array(sess), roi="valley")
    if not a.from_rust_rois:
        for ext in (".u8.npy", ".names.npy"):
            os.remove(tmp + ext)
    log("prepared", images=len(names), failed=stats["failed"], palms=len(set(ids)), sessions=sorted(set(sess)), out=a.out,
        roi="valley", valley_aligned=stats["valley_aligned"], centroid_fallback=stats["centroid_fallback"])


if __name__ == "__main__":
    main()
