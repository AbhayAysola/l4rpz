#!/usr/bin/env bash
set -euo pipefail

BINARY="${1:-./build/l4rpz}"
PASS=0
FAIL=0

check() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  expected: $(printf '%s' "$expected" | head -5)"
        echo "  actual:   $(printf '%s' "$actual"   | head -5)"
        FAIL=$((FAIL + 1))
    fi
}

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

mklz4() { lz4 -q "$1" "$2"; }

# ── multiple frames ───────────────────────────────────────────────────────────

# two frames concatenated — pattern only in second frame
printf 'AAAA' > "$tmpdir/f1a.raw"
printf 'FINDME' > "$tmpdir/f1b.raw"
mklz4 "$tmpdir/f1a.raw" "$tmpdir/f1a.lz4"
mklz4 "$tmpdir/f1b.raw" "$tmpdir/f1b.lz4"
cat "$tmpdir/f1a.lz4" "$tmpdir/f1b.lz4" > "$tmpdir/f1.lz4"
check "multi-frame: match only in second frame" \
    "$tmpdir/f1.lz4" \
    "$("$BINARY" "FINDME" "$tmpdir/f1.lz4")"

# two frames — pattern in both, count spans correctly
printf 'HIT and miss' > "$tmpdir/f2a.raw"
printf 'another HIT here' > "$tmpdir/f2b.raw"
mklz4 "$tmpdir/f2a.raw" "$tmpdir/f2a.lz4"
mklz4 "$tmpdir/f2b.raw" "$tmpdir/f2b.lz4"
cat "$tmpdir/f2a.lz4" "$tmpdir/f2b.lz4" > "$tmpdir/f2.lz4"
check "multi-frame: match in both frames, count is 2" \
    "$tmpdir/f2.lz4:2" \
    "$("$BINARY" -c "HIT" "$tmpdir/f2.lz4")"

# three frames — pattern in frames 2 and 3
printf 'frame one content' > "$tmpdir/f3a.raw"
printf 'NEEDLE in frame two' > "$tmpdir/f3b.raw"
printf 'frame three with NEEDLE' > "$tmpdir/f3c.raw"
mklz4 "$tmpdir/f3a.raw" "$tmpdir/f3a.lz4"
mklz4 "$tmpdir/f3b.raw" "$tmpdir/f3b.lz4"
mklz4 "$tmpdir/f3c.raw" "$tmpdir/f3c.lz4"
cat "$tmpdir/f3a.lz4" "$tmpdir/f3b.lz4" "$tmpdir/f3c.lz4" > "$tmpdir/f3.lz4"
check "multi-frame: three frames, count is 2" \
    "$tmpdir/f3.lz4:2" \
    "$("$BINARY" -c "NEEDLE" "$tmpdir/f3.lz4")"

# multi-block frames concatenated — pattern straddles block boundary in each frame
python3 -c "
import sys
sys.stdout.buffer.write(b'A' * (64*1024 - 3) + b'HIT' + b'B' * (64*1024))
" > "$tmpdir/f4a.raw"
python3 -c "
import sys
sys.stdout.buffer.write(b'C' * (64*1024) + b'HIT' + b'D' * (64*1024 - 3))
" > "$tmpdir/f4b.raw"
lz4 -q -B4 "$tmpdir/f4a.raw" "$tmpdir/f4a.lz4"
lz4 -q -B4 "$tmpdir/f4b.raw" "$tmpdir/f4b.lz4"
cat "$tmpdir/f4a.lz4" "$tmpdir/f4b.lz4" > "$tmpdir/f4.lz4"
check "multi-frame: multi-block frames, count is 2" \
    "$tmpdir/f4.lz4:2" \
    "$("$BINARY" -c "HIT" "$tmpdir/f4.lz4")"

# ── multiple files ────────────────────────────────────────────────────────────

# single file with match — default output is just the filename
printf 'hello world' > "$tmpdir/s1.raw"
mklz4 "$tmpdir/s1.raw" "$tmpdir/s1.lz4"
check "single file: filename printed on match" \
    "$tmpdir/s1.lz4" \
    "$("$BINARY" "hello" "$tmpdir/s1.lz4")"

# single file with no match — no output
check "single file: no output on no match" \
    "" \
    "$("$BINARY" "zzz" "$tmpdir/s1.lz4")"

# two files — only the matching one printed
printf 'MATCH here' > "$tmpdir/m1.raw"
printf 'no match here' > "$tmpdir/m2.raw"
mklz4 "$tmpdir/m1.raw" "$tmpdir/m1.lz4"
mklz4 "$tmpdir/m2.raw" "$tmpdir/m2.lz4"
check "multi-file: only matching file printed" \
    "$tmpdir/m1.lz4" \
    "$("$BINARY" "MATCH" "$tmpdir/m1.lz4" "$tmpdir/m2.lz4")"

# -L: files without matches
check "multi-file: -L prints non-matching file" \
    "$tmpdir/m2.lz4" \
    "$("$BINARY" -L "MATCH" "$tmpdir/m1.lz4" "$tmpdir/m2.lz4")"

# directory arg → finds .lz4 files recursively
mkdir -p "$tmpdir/dir/sub"
printf 'FOUND it' > "$tmpdir/dir/a.raw"
printf 'nothing here' > "$tmpdir/dir/sub/b.raw"
mklz4 "$tmpdir/dir/a.raw" "$tmpdir/dir/a.lz4"
mklz4 "$tmpdir/dir/sub/b.raw" "$tmpdir/dir/sub/b.lz4"
check "directory arg: finds .lz4 files recursively" \
    "$tmpdir/dir/a.lz4" \
    "$("$BINARY" "FOUND" "$tmpdir/dir")"

echo ""
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
