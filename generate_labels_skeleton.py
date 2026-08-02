"""
generate_labels_skeleton.py - Build a labels.csv skeleton containing only the
CLEAN-flagged tiles from tile_scores.csv (produced by preview_tiles.py), so
you're not manually cross-referencing which tiles are actually usable.

label column is left blank -- fill in 0 (stable) or 1 (eroded) per row by
eyeballing preview_contact_sheet.png, then this file is directly usable by
preprocess_tile.py --batch (run first) and train_cnn.py.

Usage:
    python preview_tiles.py --sort_by_clarity    # produces tile_scores.csv
    python generate_labels_skeleton.py
    python generate_labels_skeleton.py --include_partial   # also include PARTIAL-flagged tiles
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="tile_scores.csv")
    parser.add_argument("--tiles_dir", default="sentinel2_tiles")
    parser.add_argument("--out", default="labels_skeleton.csv")
    parser.add_argument("--include_partial", action="store_true",
                         help="Also include PARTIAL tiles, not just CLEAN")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found. Run: python preview_tiles.py --sort_by_clarity first.")

    allowed_flags = {"CLEAN"} if not args.include_partial else {"CLEAN", "PARTIAL"}

    rows_out = []
    with open(scores_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["flag"] not in allowed_flags:
                continue
            folder = row["folder"]
            # matches the naming preprocess_tile.py produces: <hash>/response.npy
            npy_relpath = f"{folder}/response.npy"
            rows_out.append({
                "filename": npy_relpath,
                "label": "",  # fill in: 0 = stable, 1 = eroded
                "location_date": row["label"],  # human-readable name @ date, for reference while labeling
                "cloud_frac": row["cloud_frac"],
            })

    if not rows_out:
        raise SystemExit("No tiles matched the flag filter -- nothing to write.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "location_date", "cloud_frac"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} tile(s) to {args.out}")
    print("Next: open it in Excel/Sheets, fill in 'label' (0=stable, 1=eroded) for each row")
    print("using preview_contact_sheet.png as reference, then drop the 'location_date' and")
    print("'cloud_frac' columns before using it with train_cnn.py (it only needs filename,label).")


if __name__ == "__main__":
    main()