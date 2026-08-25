#!/usr/bin/env python3
"""Deterministic payload corpus generator.

Two modes:
  --population : heavy-tailed realistic image population P (RQ1-RQ3)
  --hostile    : hostile-but-valid defensive set H (RQ4)

Design: everything is seeded so a run is reproducible from (seed, args) alone.
Ship the generator + manifest, NOT 50 GB of images (see plan §3). Regenerate on
the pod. Manifest is the ground truth for the RQ3 cost predictor.

Status: SKELETON. Population draw + manifest are wired; per-image encoding knobs
are stubbed with TODOs to fill in one class at a time.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, os, random
from dataclasses import dataclass, asdict

# Pillow is the reference encoder here; SIMD/opencv/vips come in at decode time (RQ5).
from PIL import Image
import numpy as np


@dataclass
class Spec:
    id: str
    format: str          # jpeg|png|webp
    width: int
    height: int
    subsampling: str     # 4:2:0|4:2:2|4:4:4|-
    progressive: bool
    icc: bool
    cmyk: bool
    exif_orient: int
    jpeg_quality: int


def draw_population(n: int, rng: random.Random) -> list[Spec]:
    """Heavy-tailed megapixels (log-normal), realistic format/chroma mix (plan §3a)."""
    specs = []
    for i in range(n):
        # megapixels ~ log-normal: median ~1.2 MP, long tail to ~48 MP
        mp = min(48.0, float(np.random.default_rng(rng.randint(0, 2**31)).lognormal(mean=0.2, sigma=0.9)))
        # derive w,h from mp at a random-ish 4:3 / 16:9 / 1:1 aspect
        aspect = rng.choice([4/3, 16/9, 1.0, 3/2])
        h = max(64, int((mp * 1e6 / aspect) ** 0.5))
        w = max(64, int(h * aspect))
        fmt = rng.choices(["jpeg", "png", "webp"], weights=[70, 15, 15])[0]
        prog = fmt == "jpeg" and rng.random() < 0.20
        sub = rng.choices(["4:2:0", "4:2:2", "4:4:4"], weights=[60, 25, 15])[0] if fmt == "jpeg" else "-"
        specs.append(Spec(
            id=f"p{i:06d}", format=fmt, width=w, height=h, subsampling=sub,
            progressive=prog, icc=rng.random() < 0.10, cmyk=rng.random() < 0.05,
            exif_orient=rng.choice([1, 1, 1, 6, 8]), jpeg_quality=rng.randint(60, 95),
        ))
    return specs


def render_and_encode(spec: Spec, rng: random.Random) -> bytes:
    """Return encoded bytes for one spec.

    TODO: honor subsampling/progressive/icc/cmyk/exif per class. For now a
    controlled synthetic image (gradient + noise) so entropy ~ quality is stable;
    swap in COCO/OpenImages samples for the 'real photo' fraction.
    """
    arr = (np.random.default_rng(rng.randint(0, 2**31)).integers(0, 256, (spec.height, spec.width, 3))).astype("uint8")
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    if spec.format == "jpeg":
        img.save(buf, "JPEG", quality=spec.jpeg_quality, progressive=spec.progressive)  # TODO subsampling, icc, cmyk, exif
    elif spec.format == "png":
        img.save(buf, "PNG")
    else:
        img.save(buf, "WEBP", quality=spec.jpeg_quality)
    return buf.getvalue()


def hostile_specs() -> list[dict]:
    """Defensive hostile-but-valid set (plan §3b). Each has a benign byte-twin.

    TODO: implement each class as a small builder returning valid bytes:
      pixel_bomb, progressive_scan_bomb, animated_webp, cmyk_fat_icc,
      png16bit, tiled_tiff, truncated, header_lie. Keep each < a few hundred KB
      on disk; the whole point is small-bytes / huge-cost.
    """
    return [
        {"class": "pixel_bomb", "note": "30000x30000 valid JPEG in ~40KB"},
        {"class": "progressive_scan_bomb", "note": "max scans, worst-case multi-pass"},
        {"class": "animated_webp", "note": "N frames, naive decoders decode all"},
        {"class": "cmyk_fat_icc", "note": "forces colorspace + profile parse"},
        {"class": "png16bit", "note": "high bit depth doubles memory"},
        {"class": "truncated", "note": "must fail cheap, not hang"},
    ]


def write_population(n: int, seed: int, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    manifest = os.path.join(out, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        cols = list(asdict(Spec("", "", 0, 0, "", False, False, False, 0, 0)).keys()) + ["bytes", "megapixels", "sha256"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for spec in draw_population(n, rng):
            data = render_and_encode(spec, rng)
            with open(os.path.join(out, f"{spec.id}.{spec.format}"), "wb") as imf:
                imf.write(data)
            row = asdict(spec)
            row["bytes"] = len(data)
            row["megapixels"] = round(spec.width * spec.height / 1e6, 3)
            row["sha256"] = hashlib.sha256(data).hexdigest()
            w.writerow(row)
    print(f"[population] {n} images + manifest -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--hostile", action="store_true")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", default="corpus_P")
    a = ap.parse_args()
    if a.population:
        write_population(a.n, a.seed, a.out)
    if a.hostile:
        for h in hostile_specs():
            print("[hostile] TODO build:", h["class"], "-", h["note"])


if __name__ == "__main__":
    main()
