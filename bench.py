#!/usr/bin/env python3
"""
Benchmark l4rpz throughput.
  alloc_ms — cudaMalloc + H2D setup
  gpu_ms   — chunk loop + drain (kernel + D2H + stream sync)
  blocks   — LZ4 block count = GPU threads launched
"""
import os
import subprocess
import tempfile

BINARY = "./build/l4rpz"
RUNS   = 5

def make_lz4(decomp_mb, pattern, density_bytes, out_path, block_flag=None):
    pat = pattern.encode() if isinstance(pattern, str) else pattern
    total = decomp_mb * 1024 * 1024
    gap = density_bytes - len(pat)
    pieces, pos, bg = [], 0, bytes(range(256))
    while pos < total:
        take = min(gap, total - pos)
        pieces.append((bg * (take // 256 + 1))[:take])
        pos += take
        if pos + len(pat) <= total:
            pieces.append(pat)
            pos += len(pat)
    raw_path = out_path + ".raw"
    with open(raw_path, 'wb') as f:
        f.write(b''.join(pieces))
    cmd = ["lz4", "-q", "-f"] + ([block_flag] if block_flag else []) + [raw_path, out_path]
    subprocess.run(cmd, check=True)
    os.unlink(raw_path)
    return os.path.getsize(out_path)

def parse_bench(stderr_text):
    alloc_ms = gpu_ms = blocks = block_kb = 0
    for line in stderr_text.splitlines():
        if not line.startswith("[bench]"):
            continue
        f = dict(kv.split("=") for kv in line.split()[1:])
        alloc_ms += float(f.get("alloc_ms", 0))
        gpu_ms += float(f.get("gpu_ms",   0))
        blocks += int(f.get("blocks",   0))
        block_kb = int(f.get("block_kb", block_kb))
    return alloc_ms, gpu_ms, blocks, block_kb

def run_bench(lz4_path, pattern_str):
    cmd = [BINARY, "--bench", "-q", pattern_str, lz4_path]
    subprocess.run(cmd, capture_output=True)          # warmup
    runs = [parse_bench(subprocess.run(cmd, capture_output=True, text=True).stderr)
            for _ in range(RUNS)]
    return sorted(runs, key=lambda x: x[1])[RUNS // 2]  # median by gpu_ms

scenarios = [
    # (decomp_mb, pat_len, density_kb, block_flag)
    (1024,  4, 64, None),    # default 4MB blocks — few threads
    (1024,  4, 64, "-B4"),   # 64KB blocks — many threads
    (1024, 16, 64, "-B4"),
    (1024,  4,  1, "-B4"),
    (1024, 16,  1, "-B4"),
    ( 256,  4, 64, "-B4"),
]

hdr = f"{'scenario':<40} {'comp MB':>7} {'blocks':>7} {'blk KB':>7} {'alloc ms':>9} {'gpu ms':>8} {'GB/s':>8}"
print(hdr)
print("-" * len(hdr))

with tempfile.TemporaryDirectory() as tmp:
    for decomp_mb, pat_len, density_kb, block_flag in scenarios:
        pattern = 'Z' * pat_len
        path = os.path.join(tmp, "bench.lz4")
        comp_sz = make_lz4(decomp_mb, pattern, density_kb * 1024, path, block_flag)

        alloc_ms, gpu_ms, blocks, block_kb = run_bench(path, pattern)
        label = f"{decomp_mb}MB  pat={pat_len}  d={density_kb}KB  {block_flag or 'default'}"
        print(f"{label:<40} {comp_sz/1024**2:>7.1f} {blocks:>7} {block_kb:>7} "
              f"{alloc_ms:>9.1f} {gpu_ms:>8.1f} {(decomp_mb/1024)/(gpu_ms/1000):>8.2f}")
