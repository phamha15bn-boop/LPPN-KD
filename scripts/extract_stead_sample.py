# Copyright (c) 2026 Qineng Song

"""Build the release sample from authorized STEAD split files."""

import argparse
import csv
import hashlib
import h5py
import json
import numpy as np
import random
from pathlib import Path

FIELDS = ["trace_name", "trace_category", "source_id", "p_arrival_sample",
          "s_arrival_sample", "split"]
SPLITS = {"train": (8, 1), "validation": (2, 0), "test": (2, 1)}
OUT = Path(__file__).resolve().parents[1] / "data" / "sample_stead"


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arrival(value):
    if isinstance(value, bytes):
        value = value.decode()
    if value is None or str(value).strip().lower() in {"", "na", "nan", "none"}:
        return -1
    number = float(value)
    return int(number) if np.isfinite(number) else -1


def select(csv_path, h5, quotas, rng, seen_names, seen_events):
    pools = {"earthquake_local": [], "noise": []}
    hits = dict.fromkeys(pools, 0)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            category = row["trace_category"].strip().lower()
            if category not in pools:
                continue
            p, s = arrival(row["p_arrival_sample"]), arrival(row["s_arrival_sample"])
            event = row["source_id"].strip()
            if category == "noise":
                if p != -1 or s != -1:
                    continue
            elif not event or not 0 <= p < s < 6000:
                continue
            hits[category] += 1
            pool = pools[category]
            candidate = (row["trace_name"], event, p, s, line)
            if len(pool) < 1024:
                pool.append(candidate)
            else:
                index = rng.randrange(hits[category])
                if index < len(pool):
                    pool[index] = candidate
    for (category, pool), count in zip(pools.items(), quotas):
        rng.shuffle(pool)
        found = 0
        for name, event, p, s, line in pool:
            if found == count:
                break
            if name in seen_names or (category != "noise" and event in seen_events):
                continue
            dataset = h5.get("data/" + name)
            if dataset is None or dataset.shape != (6000, 3) or dataset.dtype != np.float32:
                continue
            attrs = dataset.attrs
            if (attrs.get("trace_name", name) != name
                or attrs.get("trace_category") != category
                or arrival(attrs.get("p_arrival_sample")) != p
                or arrival(attrs.get("s_arrival_sample")) != s
                or (category != "noise" and attrs.get("source_id") != event)):
                continue
            wave = dataset[...]
            if not np.isfinite(wave).all():
                continue
            seen_names.add(name)
            if category != "noise":
                seen_events.add(event)
            found += 1
            yield dict(zip(FIELDS[:-1], (name, category, event, p, s))), wave, line
        if found != count:
            raise ValueError(f"{csv_path.name}: found {found} of {count} {category} traces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    rows, audit, sources = [], [], {}
    seen_names, seen_events = set(), set()
    with h5py.File(OUT / "sample.hdf5", "x") as output:
        group = output.create_group("data")
        for split, quotas in SPLITS.items():
            h5_path = args.hdf5_dir / (split + ".hdf5")
            csv_path = args.csv_dir / (split + ".csv")
            sources[split] = {"hdf5": h5_path.name, "hdf5_bytes": h5_path.stat().st_size,
                              "csv": csv_path.name}
            with h5py.File(h5_path, "r") as source:
                for row, wave, line in select(csv_path, source, quotas, rng, seen_names, seen_events):
                    row["split"] = split
                    dataset = group.create_dataset(row["trace_name"], data=wave, compression="gzip")
                    for key in FIELDS[:-1]:
                        dataset.attrs[key] = row[key]
                    rows.append(row)
                    audit.append(dict(row, source_csv_line=line, source_hdf5_key=dataset.name,
                                      waveform_sha256=hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest()))
    write_csv(OUT / "sample.csv", rows, FIELDS)
    for split in SPLITS:
        write_csv(OUT / (split + ".csv"), [r for r in rows if r["split"] == split], FIELDS)
    write_csv(OUT / "sample_manifest.csv", audit, list(audit[0]))
    with h5py.File(OUT / "sample.hdf5", "r") as h5:
        actual = set(h5["data"])
        if len(rows) != len(actual) or actual != {r["trace_name"] for r in rows}:
            raise ValueError("Sample HDF5/CSV identities differ")
    files = ["sample.hdf5", "sample.csv", "sample_manifest.csv"] + [s + ".csv" for s in SPLITS]
    manifest = {"dataset": "STEAD", "seed": 42, "sources": sources,
                "split_counts_earthquake_noise": SPLITS, "trace_count": len(rows),
                "distinct_earthquake_source_ids": len(seen_events),
                "channel_order": ["E", "N", "Z"], "sample_rate_hz": 100,
                "selection": "Seeded 1024-record reservoir per source split and category, then verified HDF5 match; unique earthquake source_id; 0 <= P < S < 6000; noise missing both arrivals normalized to -1.",
                "files": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in files}}
    with (OUT / "sample_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
