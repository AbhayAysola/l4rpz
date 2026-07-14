# l4rpz

grep but for lz4 compressed files — powered by your GPU.

Searches for a pattern in lz4 compressed files and prints the byte offsets of all matches in the decompressed content.

## Output format

```
filename:offset
```

offset is in decompressed bytes. with `-c` it's `filename:count`, with `-L` it's filename only for non-matching files.

## Features

- byte string search over lz4 compressed data
- catches patterns that straddle block boundaries
- multi-frame files supported
- requires independent blocks (lz4 default)

## Options

| flag | description |
|------|-------------|
| `-i` | case-insensitive search |
| `-c` | print match count instead of offsets |
| `-L` | print filenames with no matches (invert) |
| `-q` | quiet mode — no output, use exit code |
| `-m N` | stop after N matches per file |
| `-x` | interpret pattern as hex-encoded bytes (e.g. `0xff1122`) |
| `-V` | skip frame and block checksum verification |
| `-B` | emit per-frame timing to stderr |

## Exit codes

| code | meaning |
|------|---------|
| `0` | match found |
| `1` | no match |
| `2` | error (unreadable file, invalid lz4, GPU init failure) |

## Benchmarking
run the silesia corpus download script if running the benchmarks

`./tests/get_silesia.sh`

then run the python script

`python tests/harness.py`

## Todo

- AMD support
- better search algorithm (Boyer-Moore-Horspool)
- GPU-side checksum verification
- verbose error output
- other compression formats
