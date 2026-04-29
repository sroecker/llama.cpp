#!/usr/bin/env python3
"""Compare non-racy TCQ validation traces.

Usage:
    python3 scripts/compare_tcq_trace.py /tmp/tcq_base.bin /tmp/tcq_candidate.bin

Create traces with:
    TURBO_TCQ_DUMP_TRACE=CALLS:GROUPS \
    TURBO_TCQ_DUMP_TRACE_PATH=/tmp/tcq_base.bin \
    ./build-cuda-tcq-v10/bin/llama-bench ...
"""

import argparse
import math
import struct
import sys


HEADER = struct.Struct("<Iiiii")
META = struct.Struct("<iiiiqqqqq64s")
MAGIC = 0x31514354
VERSION = 1


def read_trace(path):
    with open(path, "rb") as f:
        header_data = f.read(HEADER.size)
        if len(header_data) != HEADER.size:
            raise ValueError(f"{path}: truncated header")

        magic, version, calls, groups, elements = HEADER.unpack(header_data)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x}")
        if version != VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        if calls < 0 or groups <= 0 or elements != 128:
            raise ValueError(f"{path}: invalid dimensions calls={calls} groups={groups} elements={elements}")

        meta = []
        for i in range(calls):
            data = f.read(META.size)
            if len(data) != META.size:
                raise ValueError(f"{path}: truncated metadata for call {i}")
            type_, is_k, captured_groups, flags, ne_total_groups, ne00, ne01, ne02, ne03, name = META.unpack(data)
            meta.append({
                "type": type_,
                "is_k": is_k,
                "captured_groups": captured_groups,
                "flags": flags,
                "ne_total_groups": ne_total_groups,
                "ne": (ne00, ne01, ne02, ne03),
                "name": name.split(b"\0", 1)[0].decode("utf-8", errors="replace"),
            })

        n_values = calls * groups * elements
        x = f.read(n_values * 4)
        if len(x) != n_values * 4:
            raise ValueError(f"{path}: truncated x payload")
        out = f.read(n_values)
        if len(out) != n_values:
            raise ValueError(f"{path}: truncated output payload")
        extra = f.read(1)
        if extra:
            raise ValueError(f"{path}: trailing data")

    return {
        "path": path,
        "calls": calls,
        "groups": groups,
        "elements": elements,
        "meta": meta,
        "x": x,
        "out": out,
    }


def compare_meta(a, b):
    if a["calls"] != b["calls"] or a["groups"] != b["groups"] or a["elements"] != b["elements"]:
        return f"dimension mismatch: {a['calls']}x{a['groups']}x{a['elements']} vs {b['calls']}x{b['groups']}x{b['elements']}"

    for i, (ma, mb) in enumerate(zip(a["meta"], b["meta"])):
        keys = ("type", "is_k", "captured_groups", "ne_total_groups", "ne", "name")
        for key in keys:
            if ma[key] != mb[key]:
                return f"metadata mismatch at call {i}, {key}: {ma[key]!r} vs {mb[key]!r}"
    return None


def find_out_mismatch(a, b):
    groups = a["groups"]
    elements = a["elements"]
    for call, meta in enumerate(a["meta"]):
        captured_groups = meta["captured_groups"]
        if captured_groups <= 0:
            continue
        start = (call * groups) * elements
        end = start + captured_groups * elements
        if a["out"][start:end] == b["out"][start:end]:
            continue

        for pos, (va, vb) in enumerate(zip(a["out"][start:end], b["out"][start:end])):
            if va != vb:
                group = pos // elements
                elem = pos % elements
                return call, group, elem, va, vb, meta["name"]
    return None


def max_x_abs_diff(a, b):
    groups = a["groups"]
    elements = a["elements"]
    max_diff = 0.0
    max_loc = None
    for call, meta in enumerate(a["meta"]):
        captured_groups = meta["captured_groups"]
        if captured_groups <= 0:
            continue
        start_value = (call * groups) * elements
        n = captured_groups * elements
        for i in range(n):
            offset = (start_value + i) * 4
            fa = struct.unpack_from("<f", a["x"], offset)[0]
            fb = struct.unpack_from("<f", b["x"], offset)[0]
            diff = abs(fa - fb)
            if max_loc is None or diff > max_diff or (math.isnan(diff) and not math.isnan(max_diff)):
                max_diff = diff
                max_loc = (call, i // elements, i % elements, fa, fb, a["meta"][call]["name"])
    return max_diff, max_loc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("candidate")
    parser.add_argument("--check-x", action="store_true", help="also report max post-FWHT normalized value difference")
    args = parser.parse_args()

    base = read_trace(args.base)
    candidate = read_trace(args.candidate)

    meta_error = compare_meta(base, candidate)
    if meta_error:
        print(f"FAIL: {meta_error}", file=sys.stderr)
        return 1

    mismatch = find_out_mismatch(base, candidate)
    if mismatch:
        call, group, elem, va, vb, name = mismatch
        print(
            f"FAIL: output mismatch at call={call} group={group} elem={elem} tensor={name}: {va} vs {vb}",
            file=sys.stderr,
        )
        return 1

    captured = sum(max(0, m["captured_groups"]) for m in base["meta"])
    print(f"PASS: output symbols match exactly for {base['calls']} calls / {captured} captured groups")

    if args.check_x:
        max_diff, loc = max_x_abs_diff(base, candidate)
        if loc is None:
            print("x max abs diff: no captured groups")
        else:
            call, group, elem, fa, fb, name = loc
            print(f"x max abs diff: {max_diff:g} at call={call} group={group} elem={elem} tensor={name} ({fa:g} vs {fb:g})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
