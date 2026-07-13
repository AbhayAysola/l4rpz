#!/usr/bin/env python3
# usage: python tests/harness.py [--bench]
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

BINARY = "./build/l4rpz"
BENCH_RUNS = 5

PASS = FAIL = 0


# helpers

def run(*args):
    r = subprocess.run([BINARY, *args], capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode # .strip for comparisons later on

def make_lz4(data: bytes, tmpdir: Path, name: str, flags=()) -> Path:
    raw = tmpdir / (name + ".raw")
    out = tmpdir / (name + ".lz4")
    raw.write_bytes(data)
    subprocess.run(["lz4", "-q", "-f", *flags, str(raw), str(out)], check=True) # -f to overwrite existing file if any
    return out

def is_uncompressed(path: Path) -> bool:
    field = int.from_bytes(path.read_bytes()[7:11], "little")
    return bool(field >> 31)

def check(desc, expected, actual):
    global PASS, FAIL
    if expected == actual:
        print(f"PASS: {desc}")
        PASS += 1
    else:
        print(f"FAIL: {desc}")
        print(f"  expected: {expected!r}")
        print(f"  actual:   {actual!r}")
        FAIL += 1

def check_exit(desc, expected_code, *args):
    check(desc, expected_code, run(*args)[2])


# checksums

def test_checksums(tmp: Path):
    base = make_lz4(b"HELLOWORLD HELLOWORLD", tmp, "base")

    # header checksums
    check_exit("header checksum: valid file accepted", 0, "HELLOWORLD", str(base))

    d = bytearray(base.read_bytes())
    d[6] ^= 0xFF
    (tmp / "bad_hc.lz4").write_bytes(d)
    check_exit("header checksum: corrupted HC byte rejected", 2, "HELLOWORLD", str(tmp / "bad_hc.lz4"))

    d = bytearray(base.read_bytes())
    d[4] ^= 0x01
    (tmp / "bad_flg.lz4").write_bytes(d)
    check_exit("header checksum: FLG tampered rejected", 2, "HELLOWORLD", str(tmp / "bad_flg.lz4"))

    # block checksums
    bx = make_lz4(b"HELLOWORLD HELLOWORLD", tmp, "bx", flags=["-BX"])
    check_exit("block checksum: valid file accepted", 0, "HELLOWORLD", str(bx))

    d = bytearray(bx.read_bytes())
    pos = 7
    pos += 4 + (int.from_bytes(d[pos:pos+4], "little") & ~(1 << 31))
    d[pos] ^= 0xFF
    (tmp / "bad_bx.lz4").write_bytes(d)
    check_exit("block checksum: corrupted block checksum rejected", 2, "HELLOWORLD", str(tmp / "bad_bx.lz4"))

    big_bx = make_lz4(b"A" * (64*1024) + b"HELLOWORLD" + b"B" * (64*1024 - 10),
                      tmp, "big_bx", flags=["-B4", "-BX"])
    check_exit("block checksum: multi block valid file accepted", 0, "HELLOWORLD", str(big_bx))

    d = bytearray(big_bx.read_bytes())
    pos = 7
    pos += 4 + (int.from_bytes(d[pos:pos+4], "little") & ~(1 << 31)) + 4  # skip block 1 + checksum
    pos += 4 + (int.from_bytes(d[pos:pos+4], "little") & ~(1 << 31))      # skip block 2, land on its checksum
    d[pos] ^= 0xFF
    (tmp / "bad_big_bx.lz4").write_bytes(d)
    check_exit("block checksum: corrupted second block checksum rejected", 2, "HELLOWORLD", str(tmp / "bad_big_bx.lz4"))

    # content checksums - TODO
    d = bytearray(base.read_bytes())
    d[-1] ^= 0xFF
    (tmp / "bad_cc.lz4").write_bytes(d)
    check_exit("content checksum: corrupted (not yet verified, should pass)", 0, "HELLOWORLD", str(tmp / "bad_cc.lz4"))


# multi frame

def test_multi_frame(tmp: Path):
    f1 = tmp / "f1.lz4"
    f1.write_bytes(make_lz4(b"AAAA", tmp, "f1a").read_bytes() +
                   make_lz4(b"FINDME", tmp, "f1b").read_bytes())
    check("multi frame: match only in second frame", f"{f1}:4", run("FINDME", str(f1))[0])

    f2 = tmp / "f2.lz4"
    f2.write_bytes(make_lz4(b"HIT and miss", tmp, "f2a").read_bytes() +
                   make_lz4(b"another HIT here", tmp, "f2b").read_bytes())
    check("multi frame: match in both frames, count is 2", f"{f2}:2", run("-c", "HIT", str(f2))[0])

    f3 = tmp / "f3.lz4"
    f3.write_bytes(make_lz4(b"frame one content", tmp, "f3a").read_bytes() +
                   make_lz4(b"NEEDLE in frame two", tmp, "f3b").read_bytes() +
                   make_lz4(b"frame three with NEEDLE", tmp, "f3c").read_bytes())
    check("multi frame: three frames, count is 2", f"{f3}:2", run("-c", "NEEDLE", str(f3))[0])

    f4 = tmp / "f4.lz4"
    f4.write_bytes(make_lz4(b"A" * (64*1024-3) + b"HIT" + b"B" * (64*1024), tmp, "f4a", flags=["-B4"]).read_bytes() +
                   make_lz4(b"C" * (64*1024)   + b"HIT" + b"D" * (64*1024-3), tmp, "f4b", flags=["-B4"]).read_bytes())
    check("multi frame: multi block frames, count is 2", f"{f4}:2", run("-c", "HIT", str(f4))[0])


# uncompressed blocks

def test_uncompressed(tmp: Path):
    data = bytearray(os.urandom(128 * 1024)) # get random data which is incompressible
    data[1000:1007] = b"PATTERN"
    rand = make_lz4(bytes(data), tmp, "rand")
    check("uncompressed block: fixture stored uncompressed", True, is_uncompressed(rand))

    check_exit("uncompressed block: match found (exit 0)", 0, "PATTERN", str(rand))
    check_exit("uncompressed block: no match (exit 1)", 1, "NOTHERE", str(rand))
    check("uncompressed block: offset printed on match", f"{rand}:1000", run("PATTERN", str(rand))[0])

    data2 = bytearray(os.urandom(128 * 1024))
    data2[-7:] = b"PATTERN"
    rand_end = make_lz4(bytes(data2), tmp, "rand_end")
    check("uncompressed block: end fixture stored uncompressed", True, is_uncompressed(rand_end))
    check_exit("uncompressed block: match at end of block", 0, "PATTERN", str(rand_end))

    multi_raw = bytearray(os.urandom(256 * 1024))
    multi_raw[1000:1007]   = b"PATTERN"
    multi_raw[70000:70007] = b"PATTERN"
    multi_unc = make_lz4(bytes(multi_raw), tmp, "multi_unc", flags=["-B4"])
    check("uncompressed multi block: count is 2", f"{multi_unc}:2", run("-c", "PATTERN", str(multi_unc))[0])
    check_exit("uncompressed multi block: no match (exit 1)", 1, "NOTHERE", str(multi_unc))


# block boundary matching

def test_block_boundary(tmp: Path):
    B = 64 * 1024  # lz4 -B4 block size

    # pattern split 1/2: last byte of block 0, first two bytes of block 1
    data = b"A" * (B - 1) + b"HIT" + b"B" * 100
    f = make_lz4(data, tmp, "seam1", flags=["-B4"])
    check_exit("block boundary: pattern split 1+2 bytes", 0, "HIT", str(f))
    check("block boundary: correct offset", f"{f}:{B - 1}", run("HIT", str(f))[0])

    # pattern split 2/1: last two bytes of block 0, first byte of block 1
    data = b"A" * (B - 2) + b"HIT" + b"B" * 100
    f = make_lz4(data, tmp, "seam2", flags=["-B4"])
    check_exit("block boundary: pattern split 2+1 bytes", 0, "HIT", str(f))
    check("block boundary: correct offset", f"{f}:{B - 2}", run("HIT", str(f))[0])

    # longer pattern straddling: only part of pattern in each block
    pat = "LONGERPATTERN"
    data = b"A" * (B - 5) + pat.encode() + b"B" * 100
    f = make_lz4(data, tmp, "seam_long", flags=["-B4"])
    check_exit("block boundary: longer pattern straddling", 0, pat, str(f))
    check("block boundary: longer pattern correct offset", f"{f}:{B - 5}", run(pat, str(f))[0])

    # no false positive: pattern not present
    data = b"A" * (B - 1) + b"XYZ" + b"B" * 100
    f = make_lz4(data, tmp, "seam_miss", flags=["-B4"])
    check_exit("block boundary: no false positive", 1, "HIT", str(f))

    # multiple seams: three blocks, patterns at both boundaries
    data = b"A" * (B - 1) + b"HIT" + b"B" * (B - 3) + b"HIT" + b"C" * 100
    f = make_lz4(data, tmp, "seam_multi", flags=["-B4"])
    check("block boundary: two seam matches", f"{f}:2", run("-c", "HIT", str(f))[0])


# flags

def test_flags(tmp: Path):
    s1 = make_lz4(b"hello world", tmp, "s1")
    check("single file: offset printed on match", f"{s1}:0", run("hello", str(s1))[0])
    check("single file: no output on no match", "", run("zzz", str(s1))[0])
    check_exit("single file: exit 0 on match", 0, "hello", str(s1))
    check_exit("single file: exit 1 on no match", 1, "zzz", str(s1))

    m1 = make_lz4(b"MATCH here", tmp, "m1")
    m2 = make_lz4(b"no match here", tmp, "m2")
    check("multi file: only matching file printed", f"{m1}:0", run("MATCH", str(m1), str(m2))[0])
    check("multi file: -L prints non-matching file", str(m2), run("-L", "MATCH", str(m1), str(m2))[0])

    (tmp / "dir" / "sub").mkdir(parents=True)
    da = make_lz4(b"FOUND it", tmp / "dir", "a")
    make_lz4(b"nothing here", tmp / "dir" / "sub", "b")
    check("directory: finds .lz4 files recursively", f"{da}:0", run("FOUND", str(tmp / "dir"))[0])

    check_exit("-q: match exits 0", 0, "-q", "hello", str(s1))
    check_exit("-q: no match exits 1", 1, "-q", "zzz", str(s1))
    check("-q: no output on match", "", run("-q", "hello", str(s1))[0])

    multi_hit = make_lz4(b"hit and hit again", tmp, "multi_hit")
    check("-c: count is 2", f"{multi_hit}:2", run("-c", "hit", str(multi_hit))[0])
    check("-i: case-insensitive match", f"{s1}:0", run("-i", "HELLO", str(s1))[0])
    check("-m 1: truncates to 1 match", f"{multi_hit}:1", run("-c", "-m", "1", "hit", str(multi_hit))[0])

    plain = tmp / "plain.txt"
    plain.write_text("hello")
    check_exit("non-lz4 file: exit 2", 2, "hello", str(plain))


# benchmarks

def _make_bench_lz4(decomp_mb, pattern, density_bytes, out_path, block_flag=None):
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
    raw = out_path + ".raw"
    with open(raw, "wb") as f:
        f.write(b"".join(pieces))
    cmd = ["lz4", "-q", "-f"] + ([block_flag] if block_flag else []) + [raw, out_path]
    subprocess.run(cmd, check=True)
    os.unlink(raw)
    return os.path.getsize(out_path)

def _parse_bench(stderr):
    alloc_ms = gpu_ms = decomp_mb = blocks = block_kb = 0
    for line in stderr.splitlines():
        if not line.startswith("[bench]"):
            continue
        f = dict(kv.split("=") for kv in line.split()[1:])
        alloc_ms += float(f.get("alloc_ms", 0))
        gpu_ms += float(f.get("gpu_ms",0))
        decomp_mb += float(f.get("decomp_mb", 0))
        blocks += int(f.get("blocks", 0))
        block_kb += int(f.get("block_kb", block_kb))
    return alloc_ms, gpu_ms, decomp_mb, blocks, block_kb

def _bench_file(path, pattern):
    cmd = [BINARY, "--bench", "-q", pattern, path]
    subprocess.run(cmd, capture_output=True)  # warmup
    runs = [_parse_bench(subprocess.run(cmd, capture_output=True, text=True).stderr)
            for _ in range(BENCH_RUNS)]
    return sorted(runs, key=lambda x: x[1])[BENCH_RUNS // 2]

def run_benchmarks():
    scenarios = [
        (1024,  4, 64, None),
        (1024,  4, 64, "-B4"),
        (1024, 16, 64, "-B4"),
        (1024,  4,  1, "-B4"),
        (1024, 16,  1, "-B4"),
        ( 257,  4, 64, "-B4"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for decomp_mb, pat_len, density_kb, block_flag in scenarios:
            pattern = "Z" * pat_len
            path = os.path.join(tmp, "bench.lz4")
            comp_sz = _make_bench_lz4(decomp_mb, pattern, density_kb * 1024, path, block_flag)
            alloc_ms, gpu_ms, _, blocks, block_kb = _bench_file(path, pattern)
            label = f"{decomp_mb}MB pat={pat_len} d={density_kb}KB {block_flag or 'default'}"
            gbs = (decomp_mb / 1024) / (gpu_ms / 1000)
            print(f"{label}: {comp_sz/1024**2:.1f}MB comp  {blocks} blocks  {alloc_ms:.0f}ms alloc  {gpu_ms:.0f}ms gpu  {gbs:.2f} GB/s")

def run_silesia_benchmarks():
    for label, directory in [("default blocks", "data/silesia_lz4"),
                              ("64KB blocks",    "data/silesia_lz4_b4")]:
        files = sorted(Path(directory).glob("*.lz4"))
        if not files:
            continue
        print(f"\nsilesia ({label})")
        total_mb = total_ms = 0.0
        for f in files:
            _, gpu_ms, decomp_mb, blocks, block_kb = _bench_file(str(f), "the")
            total_mb += decomp_mb
            total_ms += gpu_ms
            print(f"  {f.name}: {decomp_mb:.1f}MB  {blocks} blocks  {gpu_ms:.0f}ms  {decomp_mb/(gpu_ms/1000):.2f} GB/s")
        print(f"  total: {total_mb:.1f}MB  {total_ms:.0f}ms  {total_mb/(total_ms/1000):.2f} GB/s")


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench",   action="store_true", help="run synthetic benchmarks")
    parser.add_argument("--silesia", action="store_true", help="run silesia benchmarks (requires data/silesia_lz4/)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_checksums(tmp)
        test_multi_frame(tmp)
        test_uncompressed(tmp)
        test_block_boundary(tmp)
        test_flags(tmp)

    print(f"\n{PASS} passed, {FAIL} failed")

    if args.bench:
        run_benchmarks()
    if args.silesia:
        run_silesia_benchmarks()

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
